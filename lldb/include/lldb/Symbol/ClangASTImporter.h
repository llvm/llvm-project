//===-- ClangASTImporter.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangASTImporter_h_
#define liblldb_ClangASTImporter_h_

#include <map>

#include "lldb/lldb-types.h"

#include "clang/AST/ASTImporter.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "lldb/Symbol/ClangNamespaceDecl.h"

namespace lldb_private {

class ClangASTImporter 
{
public:
    ClangASTImporter () :
        m_file_manager(clang::FileSystemOptions())
    {
    }
    
    clang::QualType
    CopyType (clang::ASTContext *dst_ctx,
              clang::ASTContext *src_ctx,
              clang::QualType type);
    
    lldb::clang_type_t
    CopyType (clang::ASTContext *dst_ctx,
              clang::ASTContext *src_ctx,
              lldb::clang_type_t type);
    
    clang::Decl *
    CopyDecl (clang::ASTContext *dst_ctx,
              clang::ASTContext *src_ctx,
              clang::Decl *decl);
    
    lldb::clang_type_t
    DeportType (clang::ASTContext *dst_ctx,
                clang::ASTContext *src_ctx,
                lldb::clang_type_t type);
    
    clang::Decl *
    DeportDecl (clang::ASTContext *dst_ctx,
                clang::ASTContext *src_ctx,
                clang::Decl *decl);
    
    void
    CompleteDecl (clang::Decl *decl);
        
    bool
    CompleteTagDecl (clang::TagDecl *decl);
    
    bool
    CompleteTagDeclWithOrigin (clang::TagDecl *decl, clang::TagDecl *origin);
    
    bool
    CompleteObjCInterfaceDecl (clang::ObjCInterfaceDecl *interface_decl);
    
    bool
    ResolveDeclOrigin (const clang::Decl *decl, clang::Decl **original_decl, clang::ASTContext **original_ctx)
    {
        DeclOrigin origin = GetDeclOrigin(decl);
        
        if (original_decl)
            *original_decl = origin.decl;
        
        if (original_ctx)
            *original_ctx = origin.ctx;
        
        return origin.Valid();
    }
    
    void
    SetDeclOrigin (const clang::Decl *decl, clang::Decl *original_decl);
    
    ClangASTMetadata *
    GetDeclMetadata (const clang::Decl *decl);
    
    //
    // Namespace maps
    //
    
    typedef std::vector < std::pair<lldb::ModuleSP, ClangNamespaceDecl> > NamespaceMap;
    typedef STD_SHARED_PTR(NamespaceMap) NamespaceMapSP;
    
    void RegisterNamespaceMap (const clang::NamespaceDecl *decl, 
                               NamespaceMapSP &namespace_map);
                           
    NamespaceMapSP GetNamespaceMap (const clang::NamespaceDecl *decl);
    
    void BuildNamespaceMap (const clang::NamespaceDecl *decl);
    
    //
    // Comleters for maps
    //
    
    class MapCompleter 
    {
    public:
        virtual ~MapCompleter ();
        
        virtual void CompleteNamespaceMap (NamespaceMapSP &namespace_map,
                                           const ConstString &name,
                                           NamespaceMapSP &parent_map) const = 0;
    };
    
    void InstallMapCompleter (clang::ASTContext *dst_ctx, MapCompleter &completer)
    {
        ASTContextMetadataSP context_md;
        ContextMetadataMap::iterator context_md_iter = m_metadata_map.find(dst_ctx);
        
        if (context_md_iter == m_metadata_map.end())
        {
            context_md = ASTContextMetadataSP(new ASTContextMetadata(dst_ctx));
            m_metadata_map[dst_ctx] = context_md;
        }
        else
        {
            context_md = context_md_iter->second;
        }
                
        context_md->m_map_completer = &completer;
    }
    
    void ForgetDestination (clang::ASTContext *dst_ctx);
    void ForgetSource (clang::ASTContext *dst_ctx, clang::ASTContext *src_ctx);
private:
    struct DeclOrigin 
    {
        DeclOrigin () :
            ctx(NULL),
            decl(NULL)
        {
        }
        
        DeclOrigin (clang::ASTContext *_ctx,
                    clang::Decl *_decl) :
            ctx(_ctx),
            decl(_decl)
        {
        }
        
        DeclOrigin (const DeclOrigin &rhs)
        {
            ctx = rhs.ctx;
            decl = rhs.decl;
        }
        
        void operator= (const DeclOrigin &rhs)
        {
            ctx = rhs.ctx;
            decl = rhs.decl;
        }
        
        bool 
        Valid ()
        {
            return (ctx != NULL || decl != NULL);
        }
        
        clang::ASTContext  *ctx;
        clang::Decl        *decl;
    };
    
    typedef std::map<const clang::Decl *, DeclOrigin>   OriginMap;
    
    class Minion : public clang::ASTImporter
    {
    public:
        Minion (ClangASTImporter &master,
                clang::ASTContext *target_ctx,
                clang::ASTContext *source_ctx) :
            clang::ASTImporter(*target_ctx,
                               master.m_file_manager,
                               *source_ctx,
                               master.m_file_manager,
                               true /*minimal*/),
            m_master(master),
            m_source_ctx(source_ctx)
        {
        }
        
        void ImportDefinitionTo (clang::Decl *to, clang::Decl *from);
        
        clang::Decl *Imported (clang::Decl *from, clang::Decl *to);
        
        ClangASTImporter   &m_master;
        clang::ASTContext  *m_source_ctx;
    };
    
    typedef STD_SHARED_PTR(Minion) MinionSP;
    typedef std::map<clang::ASTContext *, MinionSP> MinionMap;
    typedef std::map<const clang::NamespaceDecl *, NamespaceMapSP> NamespaceMetaMap;
    
    struct ASTContextMetadata
    {
        ASTContextMetadata(clang::ASTContext *dst_ctx) :
            m_dst_ctx (dst_ctx),
            m_minions (),
            m_origins (),
            m_namespace_maps (),
            m_map_completer (NULL)
        {
        }
        
        clang::ASTContext      *m_dst_ctx;
        MinionMap               m_minions;
        OriginMap               m_origins;
        
        NamespaceMetaMap        m_namespace_maps;
        MapCompleter           *m_map_completer;
    };
    
    typedef STD_SHARED_PTR(ASTContextMetadata) ASTContextMetadataSP;    
    typedef std::map<const clang::ASTContext *, ASTContextMetadataSP> ContextMetadataMap;
    
    ContextMetadataMap m_metadata_map;
    
    ASTContextMetadataSP
    GetContextMetadata (clang::ASTContext *dst_ctx)
    {
        ContextMetadataMap::iterator context_md_iter = m_metadata_map.find(dst_ctx);
        
        if (context_md_iter == m_metadata_map.end())
        {
            ASTContextMetadataSP context_md = ASTContextMetadataSP(new ASTContextMetadata(dst_ctx));
            m_metadata_map[dst_ctx] = context_md;
            return context_md;
        }
        else
        {
            return context_md_iter->second;
        }
    }
    
    ASTContextMetadataSP
    MaybeGetContextMetadata (clang::ASTContext *dst_ctx)
    {
        ContextMetadataMap::iterator context_md_iter = m_metadata_map.find(dst_ctx);

        if (context_md_iter != m_metadata_map.end())
            return context_md_iter->second;
        else
            return ASTContextMetadataSP();
    }
    
    MinionSP
    GetMinion (clang::ASTContext *dst_ctx, clang::ASTContext *src_ctx)
    {
        ASTContextMetadataSP context_md = GetContextMetadata(dst_ctx);
        
        MinionMap &minions = context_md->m_minions;
        MinionMap::iterator minion_iter = minions.find(src_ctx);
        
        if (minion_iter == minions.end())
        {
            MinionSP minion = MinionSP(new Minion(*this, dst_ctx, src_ctx));
            minions[src_ctx] = minion;
            return minion;
        }
        else
        {
            return minion_iter->second;
        }       
    }
    
    DeclOrigin
    GetDeclOrigin (const clang::Decl *decl);
        
    clang::FileManager      m_file_manager;
};
    
}

#endif
