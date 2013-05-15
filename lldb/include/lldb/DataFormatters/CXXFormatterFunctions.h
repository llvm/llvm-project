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
#include <time.h>

#include "lldb/lldb-forward.h"

#include "lldb/Core/ConstString.h"
#include "lldb/DataFormatters/FormatClasses.h"
#include "lldb/Target/Target.h"

#include "clang/AST/ASTContext.h"

namespace lldb_private {
    namespace formatters
    {
        bool
        ExtractValueFromObjCExpression (ValueObject &valobj,
                                        const char* target_type,
                                        const char* selector,
                                        uint64_t &value);
        
        bool
        ExtractSummaryFromObjCExpression (ValueObject &valobj,
                                          const char* target_type,
                                          const char* selector,
                                          Stream &stream);

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
        
        size_t
        ExtractIndexFromString (const char* item_name);
        
        time_t
        GetOSXEpoch ();
        
        bool
        Char16StringSummaryProvider (ValueObject& valobj, Stream& stream); // char16_t* and unichar*
        
        bool
        Char32StringSummaryProvider (ValueObject& valobj, Stream& stream); // char32_t*
        
        bool
        WCharStringSummaryProvider (ValueObject& valobj, Stream& stream); // wchar_t*
        
        bool
        Char16SummaryProvider (ValueObject& valobj, Stream& stream); // char16_t and unichar
        
        bool
        Char32SummaryProvider (ValueObject& valobj, Stream& stream); // char32_t
        
        bool
        WCharSummaryProvider (ValueObject& valobj, Stream& stream); // wchar_t
        
        bool
        LibcxxStringSummaryProvider (ValueObject& valobj, Stream& stream); // libc++ std::string

        bool
        LibcxxWStringSummaryProvider (ValueObject& valobj, Stream& stream); // libc++ std::wstring
        
        bool
        ObjCClassSummaryProvider (ValueObject& valobj, Stream& stream);
        
        SyntheticChildrenFrontEnd* ObjCClassSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        template<bool name_entries>
        bool
        NSDictionarySummaryProvider (ValueObject& valobj, Stream& stream);
        
        bool
        NSIndexSetSummaryProvider (ValueObject& valobj, Stream& stream);
        
        bool
        NSArraySummaryProvider (ValueObject& valobj, Stream& stream);
        
        template<bool cf_style>
        bool
        NSSetSummaryProvider (ValueObject& valobj, Stream& stream);
        
        template<bool needs_at>
        bool
        NSDataSummaryProvider (ValueObject& valobj, Stream& stream);
        
        bool
        NSNumberSummaryProvider (ValueObject& valobj, Stream& stream);

        bool
        NSNotificationSummaryProvider (ValueObject& valobj, Stream& stream);
        
        bool
        NSTimeZoneSummaryProvider (ValueObject& valobj, Stream& stream);
        
        bool
        NSMachPortSummaryProvider (ValueObject& valobj, Stream& stream);
        
        bool
        CFBagSummaryProvider (ValueObject& valobj, Stream& stream);
        
        bool
        CFBinaryHeapSummaryProvider (ValueObject& valobj, Stream& stream);
        
        bool
        CFBitVectorSummaryProvider (ValueObject& valobj, Stream& stream);
        
        bool
        NSDateSummaryProvider (ValueObject& valobj, Stream& stream);
        
        bool
        CFAbsoluteTimeSummaryProvider (ValueObject& valobj, Stream& stream);
        
        bool
        NSBundleSummaryProvider (ValueObject& valobj, Stream& stream);
        
        bool
        NSStringSummaryProvider (ValueObject& valobj, Stream& stream);
        
        bool
        NSAttributedStringSummaryProvider (ValueObject& valobj, Stream& stream);
        
        bool
        NSMutableAttributedStringSummaryProvider (ValueObject& valobj, Stream& stream);
        
        bool
        NSURLSummaryProvider (ValueObject& valobj, Stream& stream);
        
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
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
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
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
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
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
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
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~NSDictionaryISyntheticFrontEnd ();
        private:
            ExecutionContextRef m_exe_ctx_ref;
            uint8_t m_ptr_size;
            lldb::ByteOrder m_order;
            DataDescriptor_32 *m_data_32;
            DataDescriptor_64 *m_data_64;
            lldb::addr_t m_data_ptr;
            ClangASTType m_pair_type;
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
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~NSDictionaryMSyntheticFrontEnd ();
        private:
            ExecutionContextRef m_exe_ctx_ref;
            uint8_t m_ptr_size;
            lldb::ByteOrder m_order;
            DataDescriptor_32 *m_data_32;
            DataDescriptor_64 *m_data_64;
            ClangASTType m_pair_type;
            std::vector<DictionaryItemDescriptor> m_children;
        };
        
        class NSDictionaryCodeRunningSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            NSDictionaryCodeRunningSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~NSDictionaryCodeRunningSyntheticFrontEnd ();
        };
        
        SyntheticChildrenFrontEnd* NSDictionarySyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        class NSSetISyntheticFrontEnd : public SyntheticChildrenFrontEnd
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
            
            struct SetItemDescriptor
            {
                lldb::addr_t item_ptr;
                lldb::ValueObjectSP valobj_sp;
            };
            
        public:
            NSSetISyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~NSSetISyntheticFrontEnd ();
        private:
            ExecutionContextRef m_exe_ctx_ref;
            uint8_t m_ptr_size;
            DataDescriptor_32 *m_data_32;
            DataDescriptor_64 *m_data_64;
            lldb::addr_t m_data_ptr;
            std::vector<SetItemDescriptor> m_children;
        };
        
        class NSOrderedSetSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        private:
            
        public:
            NSOrderedSetSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~NSOrderedSetSyntheticFrontEnd ();
        private:
            uint32_t m_count;
            std::map<uint32_t,lldb::ValueObjectSP> m_children;
        };
        
        class NSSetMSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        private:
            struct DataDescriptor_32
            {
                uint32_t _used : 26;
                uint32_t _size;
                uint32_t _mutations;
                uint32_t _objs_addr;
            };
            struct DataDescriptor_64
            {
                uint64_t _used : 58;
                uint64_t _size;
                uint64_t _mutations;
                uint64_t _objs_addr;
            };
            struct SetItemDescriptor
            {
                lldb::addr_t item_ptr;
                lldb::ValueObjectSP valobj_sp;
            };
        public:
            NSSetMSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~NSSetMSyntheticFrontEnd ();
        private:
            ExecutionContextRef m_exe_ctx_ref;
            uint8_t m_ptr_size;
            DataDescriptor_32 *m_data_32;
            DataDescriptor_64 *m_data_64;
            std::vector<SetItemDescriptor> m_children;
        };
                
        class NSSetCodeRunningSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            NSSetCodeRunningSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~NSSetCodeRunningSyntheticFrontEnd ();
        };
        
        SyntheticChildrenFrontEnd* NSSetSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        class LibcxxVectorBoolSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            LibcxxVectorBoolSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~LibcxxVectorBoolSyntheticFrontEnd ();
        private:
            ExecutionContextRef m_exe_ctx_ref;
            uint64_t m_count;
            lldb::addr_t m_base_data_address;
            EvaluateExpressionOptions m_options;
        };
        
        SyntheticChildrenFrontEnd* LibcxxVectorBoolSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        bool
        LibcxxContainerSummaryProvider (ValueObject& valobj, Stream& stream);
        
        class LibstdcppVectorBoolSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            LibstdcppVectorBoolSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~LibstdcppVectorBoolSyntheticFrontEnd ();
        private:
            ExecutionContextRef m_exe_ctx_ref;
            uint64_t m_count;
            lldb::addr_t m_base_data_address;
            EvaluateExpressionOptions m_options;
        };
        
        SyntheticChildrenFrontEnd* LibstdcppVectorBoolSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        class LibstdcppMapIteratorSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            LibstdcppMapIteratorSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~LibstdcppMapIteratorSyntheticFrontEnd ();
        private:
            ExecutionContextRef m_exe_ctx_ref;
            lldb::addr_t m_pair_address;
            ClangASTType m_pair_type;
            EvaluateExpressionOptions m_options;
            lldb::ValueObjectSP m_pair_sp;
        };
        
        SyntheticChildrenFrontEnd* LibstdcppMapIteratorSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        class LibCxxMapIteratorSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            LibCxxMapIteratorSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~LibCxxMapIteratorSyntheticFrontEnd ();
        private:
            ValueObject *m_pair_ptr;
        };
        
        SyntheticChildrenFrontEnd* LibCxxMapIteratorSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);

        class VectorIteratorSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            VectorIteratorSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp,
                                             ConstString item_name);
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~VectorIteratorSyntheticFrontEnd ();
        private:
            ExecutionContextRef m_exe_ctx_ref;
            ConstString m_item_name;
            lldb::ValueObjectSP m_item_sp;
        };
        
        SyntheticChildrenFrontEnd* LibCxxVectorIteratorSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        SyntheticChildrenFrontEnd* LibStdcppVectorIteratorSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        class LibcxxSharedPtrSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            LibcxxSharedPtrSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~LibcxxSharedPtrSyntheticFrontEnd ();
        private:
            ValueObject* m_cntrl;
            lldb::ValueObjectSP m_count_sp;
            lldb::ValueObjectSP m_weak_count_sp;
            uint8_t m_ptr_size;
            lldb::ByteOrder m_byte_order;
        };
        
        SyntheticChildrenFrontEnd* LibcxxSharedPtrSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        class LibcxxStdVectorSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            LibcxxStdVectorSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~LibcxxStdVectorSyntheticFrontEnd ();
        private:
            ValueObject* m_start;
            ValueObject* m_finish;
            ClangASTType m_element_type;
            uint32_t m_element_size;
            std::map<size_t,lldb::ValueObjectSP> m_children;
        };
        
        SyntheticChildrenFrontEnd* LibcxxStdVectorSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        class LibcxxStdListSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            LibcxxStdListSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~LibcxxStdListSyntheticFrontEnd ();
        private:
            bool
            HasLoop();
            
            size_t m_list_capping_size;
            static const bool g_use_loop_detect = true;
            lldb::addr_t m_node_address;
            ValueObject* m_head;
            ValueObject* m_tail;
            ClangASTType m_element_type;
            size_t m_count;
            std::map<size_t,lldb::ValueObjectSP> m_children;
        };
        
        SyntheticChildrenFrontEnd* LibcxxStdListSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        class LibcxxStdMapSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            LibcxxStdMapSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~LibcxxStdMapSyntheticFrontEnd ();
        private:
            bool
            GetDataType();
            
            void
            GetValueOffset (const lldb::ValueObjectSP& node);
            
            ValueObject* m_tree;
            ValueObject* m_root_node;
            ClangASTType m_element_type;
            uint32_t m_skip_size;
            size_t m_count;
            std::map<size_t,lldb::ValueObjectSP> m_children;
        };
        
        SyntheticChildrenFrontEnd* LibcxxStdMapSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
    } // namespace formatters
} // namespace lldb_private

#endif // liblldb_CXXFormatterFunctions_h_
