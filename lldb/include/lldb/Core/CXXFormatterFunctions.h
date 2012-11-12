//===-- CXXFormatterFunctions.h------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CXXFormatterFunctions_h_
#define liblldb_CXXFormatterFunctions_h_

#include <stdint.h>
#include "lldb/lldb-forward.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/FormatClasses.h"

#include "clang/AST/ASTContext.h"

namespace lldb_private {
    namespace formatters
    {
        
        bool
        ExtractValueFromObjCExpression (ValueObject &valobj,
                                        const char* target_type,
                                        const char* selector,
                                        uint64_t &value);
        
        lldb::ValueObjectSP
        CallSelectorOnObject (ValueObject &valobj,
                              const char* return_type,
                              const char* selector,
                              uint64_t index);
        
        lldb::ValueObjectSP
        CallSelectorOnObject (ValueObject &valobj,
                              const char* return_type,
                              const char* selector,
                              const char* key);
        
        template<bool name_entries>
        bool
        NSDictionarySummaryProvider (ValueObject& valobj, Stream& stream);
        
        bool
        NSArraySummaryProvider (ValueObject& valobj, Stream& stream);
        
        template<bool needs_at>
        bool
        NSDataSummaryProvider (ValueObject& valobj, Stream& stream);
        
        bool
        NSNumberSummaryProvider (ValueObject& valobj, Stream& stream);

        bool
        NSStringSummaryProvider (ValueObject& valobj, Stream& stream);
        
        bool
        ObjCBOOLSummaryProvider (ValueObject& valobj, Stream& stream);
        
        template <bool is_sel_ptr>
        bool
        ObjCSELSummaryProvider (ValueObject& valobj, Stream& stream);
        
        bool
        RuntimeSpecificDescriptionSummaryProvider (ValueObject& valobj, Stream& stream);
        
        extern template bool
        NSDictionarySummaryProvider<true> (ValueObject&, Stream&) ;
        
        extern template bool
        NSDictionarySummaryProvider<false> (ValueObject&, Stream&) ;
        
        extern template bool
        NSDataSummaryProvider<true> (ValueObject&, Stream&) ;
        
        extern template bool
        NSDataSummaryProvider<false> (ValueObject&, Stream&) ;
        
        extern template bool
        ObjCSELSummaryProvider<true> (ValueObject&, Stream&);

        extern template bool
        ObjCSELSummaryProvider<false> (ValueObject&, Stream&);
        
        class NSArrayMSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        private:
            struct DataDescriptor_32
            {
                uint32_t _used;
                uint32_t _priv1 : 2 ;
                uint32_t _size : 30;
                uint32_t _priv2 : 2;
                uint32_t offset : 30;
                uint32_t _priv3;
                uint32_t _data;
            };
            struct DataDescriptor_64
            {
                uint64_t _used;
                uint64_t _priv1 : 2 ;
                uint64_t _size : 62;
                uint64_t _priv2 : 2;
                uint64_t offset : 62;
                uint32_t _priv3;
                uint64_t _data;
            };
        public:
            NSArrayMSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual uint32_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (uint32_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual uint32_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~NSArrayMSyntheticFrontEnd ();
        private:
            ExecutionContextRef m_exe_ctx_ref;
            uint8_t m_ptr_size;
            DataDescriptor_32 *m_data_32;
            DataDescriptor_64 *m_data_64;
            ClangASTType m_id_type;
            std::vector<lldb::ValueObjectSP> m_children;
        };
        
        class NSArrayISyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            NSArrayISyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual uint32_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (uint32_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual uint32_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~NSArrayISyntheticFrontEnd ();
        private:
            ExecutionContextRef m_exe_ctx_ref;
            uint8_t m_ptr_size;
            uint64_t m_items;
            lldb::addr_t m_data_ptr;
            ClangASTType m_id_type;
            std::vector<lldb::ValueObjectSP> m_children;
        };
        
        class NSArrayCodeRunningSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            NSArrayCodeRunningSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual uint32_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (uint32_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual uint32_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~NSArrayCodeRunningSyntheticFrontEnd ();
        };
        
        SyntheticChildrenFrontEnd* NSArraySyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        class NSDictionaryISyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        private:
            struct DataDescriptor_32
            {
                uint32_t _used : 26;
                uint32_t _szidx : 6;
            };
            struct DataDescriptor_64
            {
                uint64_t _used : 58;
                uint32_t _szidx : 6;
            };
            
            struct DictionaryItemDescriptor
            {
                lldb::addr_t key_ptr;
                lldb::addr_t val_ptr;
                lldb::ValueObjectSP valobj_sp;
            };
            
        public:
            NSDictionaryISyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual uint32_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (uint32_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual uint32_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~NSDictionaryISyntheticFrontEnd ();
        private:
            ExecutionContextRef m_exe_ctx_ref;
            uint8_t m_ptr_size;
            DataDescriptor_32 *m_data_32;
            DataDescriptor_64 *m_data_64;
            lldb::addr_t m_data_ptr;
            std::vector<DictionaryItemDescriptor> m_children;
        };
        
        class NSDictionaryMSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        private:
            struct DataDescriptor_32
            {
                uint32_t _used : 26;
                uint32_t _kvo : 1;
                uint32_t _size;
                uint32_t _mutations;
                uint32_t _objs_addr;
                uint32_t _keys_addr;
            };
            struct DataDescriptor_64
            {
                uint64_t _used : 58;
                uint32_t _kvo : 1;
                uint64_t _size;
                uint64_t _mutations;
                uint64_t _objs_addr;
                uint64_t _keys_addr;
            };
            struct DictionaryItemDescriptor
            {
                lldb::addr_t key_ptr;
                lldb::addr_t val_ptr;
                lldb::ValueObjectSP valobj_sp;
            };
        public:
            NSDictionaryMSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual uint32_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (uint32_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual uint32_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~NSDictionaryMSyntheticFrontEnd ();
        private:
            ExecutionContextRef m_exe_ctx_ref;
            uint8_t m_ptr_size;
            uint64_t m_items;
            DataDescriptor_32 *m_data_32;
            DataDescriptor_64 *m_data_64;
            std::vector<DictionaryItemDescriptor> m_children;
        };
        
        class NSDictionaryCodeRunningSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            NSDictionaryCodeRunningSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual uint32_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (uint32_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual uint32_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~NSDictionaryCodeRunningSyntheticFrontEnd ();
        };
        
        SyntheticChildrenFrontEnd* NSDictionarySyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
    }
}

#endif
