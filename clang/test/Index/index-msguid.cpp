template <class T, const _GUID &ID = __uuidof(T)>
class GUIDHolder {
public:
  virtual ~GUIDHolder() {}
};

class  __declspec(uuid("12345678-1234-5678-ABCD-12345678ABCD")) GUIDInterface {};

class GUIDUse : public GUIDHolder<GUIDInterface> {
  ~GUIDUse() {}
  // CHECK: RelOver | ~GUIDHolder | c:@S@GUIDHolder>#$@S@GUIDInterface#@MG@GUID{12345678-1234-5678-abcd-12345678abcd}@F@~GUIDHolder#
};

// RUN: c-index-test core -print-source-symbols -- -std=c++11 -fms-extensions %s | FileCheck %s
