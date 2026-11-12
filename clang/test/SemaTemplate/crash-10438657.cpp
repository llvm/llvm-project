// RUN: not %clang_cc1 -fsyntax-only %s 2> %t
// RUN: FileCheck %s < %t
// CHECK: 10 errors
template<typename _CharT>
class collate : public locale::facet {

protected:
virtual ~collate() {}
  class wxObject;
  class __attribute__ ((visibility("default"))) wxGDIRefData 
    : public wxObjectRefData {};
  class __attribute__ ((visibility("default"))) wxGDIObject : public wxObject { \
      public:
      virtual bool IsOk() const {
        return m_refData && static_cast<wxGDIRefData *>(m_refData)->IsOk(); 
