//===-- ClangExternalASTSourceCallbacks.h -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangExternalASTSourceCallbacks_h_
#define liblldb_ClangExternalASTSourceCallbacks_h_

// C Includes
// C++ Includes
#include <string>
#include <vector>
#include <stdint.h>

// Other libraries and framework includes
#include "clang/AST/CharUnits.h"

// Project includes
#include "lldb/lldb-enumerations.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/ClangExternalASTSourceCommon.h"

namespace lldb_private {

class ClangExternalASTSourceCallbacks : public ClangExternalASTSourceCommon
{
public:

    typedef void (*CompleteTagDeclCallback)(void *baton, clang::TagDecl *);
    typedef void (*CompleteObjCInterfaceDeclCallback)(void *baton, clang::ObjCInterfaceDecl *);
    typedef void (*FindExternalVisibleDeclsByNameCallback)(void *baton, const clang::DeclContext *DC, clang::DeclarationName Name, llvm::SmallVectorImpl <clang::NamedDecl *> *results);
    typedef bool (*LayoutRecordTypeCallback)(void *baton, 
                                             const clang::RecordDecl *Record,
                                             uint64_t &Size, 
                                             uint64_t &Alignment,
                                             llvm::DenseMap <const clang::FieldDecl *, uint64_t> &FieldOffsets,
                                             llvm::DenseMap <const clang::CXXRecordDecl *, clang::CharUnits> &BaseOffsets,
                                             llvm::DenseMap <const clang::CXXRecordDecl *, clang::CharUnits> &VirtualBaseOffsets);

    ClangExternalASTSourceCallbacks (CompleteTagDeclCallback tag_decl_callback,
                                     CompleteObjCInterfaceDeclCallback objc_decl_callback,
                                     FindExternalVisibleDeclsByNameCallback find_by_name_callback,
                                     LayoutRecordTypeCallback layout_record_type_callback,
                                     void *callback_baton) :
        m_callback_tag_decl (tag_decl_callback),
        m_callback_objc_decl (objc_decl_callback),
        m_callback_find_by_name (find_by_name_callback),
        m_callback_layout_record_type (layout_record_type_callback),
        m_callback_baton (callback_baton)
    {
    }
    
    //------------------------------------------------------------------
    // clang::ExternalASTSource
    //------------------------------------------------------------------

    virtual clang::Decl *
    GetExternalDecl (uint32_t ID)
    {
        // This method only needs to be implemented if the AST source ever
        // passes back decl sets as VisibleDeclaration objects.
        return 0; 
    }
    
    virtual clang::Stmt *
    GetExternalDeclStmt (uint64_t Offset)
    {
        // This operation is meant to be used via a LazyOffsetPtr.  It only
        // needs to be implemented if the AST source uses methods like
        // FunctionDecl::setLazyBody when building decls.
        return 0; 
    }
	
    virtual clang::Selector 
    GetExternalSelector (uint32_t ID)
    {
        // This operation only needs to be implemented if the AST source
        // returns non-zero for GetNumKnownSelectors().
        return clang::Selector();
    }

	virtual uint32_t
    GetNumExternalSelectors()
    {
        return 0;
    }
    
    virtual clang::CXXBaseSpecifier *
    GetExternalCXXBaseSpecifiers(uint64_t Offset)
    {
        return NULL; 
    }
	
    virtual void 
    MaterializeVisibleDecls (const clang::DeclContext *decl_ctx)
    {
        return;
    }
	
	virtual clang::ExternalLoadResult 
    FindExternalLexicalDecls (const clang::DeclContext *decl_ctx,
                              bool (*isKindWeWant)(clang::Decl::Kind),
                              llvm::SmallVectorImpl<clang::Decl*> &decls)
    {
        // This is used to support iterating through an entire lexical context,
        // which isn't something the debugger should ever need to do.
        return clang::ELR_Failure;
    }
    
    virtual bool
    FindExternalVisibleDeclsByName (const clang::DeclContext *decl_ctx,
                                    clang::DeclarationName decl_name);
    
    virtual void
    CompleteType (clang::TagDecl *tag_decl);
    
    virtual void
    CompleteType (clang::ObjCInterfaceDecl *objc_decl);
    
    bool 
    layoutRecordType(const clang::RecordDecl *Record,
                     uint64_t &Size, 
                     uint64_t &Alignment,
                     llvm::DenseMap <const clang::FieldDecl *, uint64_t> &FieldOffsets,
                     llvm::DenseMap <const clang::CXXRecordDecl *, clang::CharUnits> &BaseOffsets,
                     llvm::DenseMap <const clang::CXXRecordDecl *, clang::CharUnits> &VirtualBaseOffsets);
    void
    SetExternalSourceCallbacks (CompleteTagDeclCallback tag_decl_callback,
                                CompleteObjCInterfaceDeclCallback objc_decl_callback,
                                FindExternalVisibleDeclsByNameCallback find_by_name_callback,
                                LayoutRecordTypeCallback layout_record_type_callback,
                                void *callback_baton)
    {
        m_callback_tag_decl = tag_decl_callback;
        m_callback_objc_decl = objc_decl_callback;
        m_callback_find_by_name = find_by_name_callback;
        m_callback_layout_record_type = layout_record_type_callback;
        m_callback_baton = callback_baton;    
    }

    void
    RemoveExternalSourceCallbacks (void *callback_baton)
    {
        if (callback_baton == m_callback_baton)
        {
            m_callback_tag_decl = NULL;
            m_callback_objc_decl = NULL;
            m_callback_find_by_name = NULL;
            m_callback_layout_record_type = NULL;
        }
    }

protected:
    //------------------------------------------------------------------
    // Classes that inherit from ClangExternalASTSourceCallbacks can see and modify these
    //------------------------------------------------------------------
    CompleteTagDeclCallback                 m_callback_tag_decl;
    CompleteObjCInterfaceDeclCallback       m_callback_objc_decl;
    FindExternalVisibleDeclsByNameCallback  m_callback_find_by_name;
    LayoutRecordTypeCallback                m_callback_layout_record_type;
    void *                                  m_callback_baton;
};

} // namespace lldb_private

#endif  // liblldb_ClangExternalASTSourceCallbacks_h_
