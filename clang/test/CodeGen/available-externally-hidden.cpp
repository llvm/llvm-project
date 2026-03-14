// RUN: %clang_cc1 -O2 -fvisibility=hidden -std=c++11 -emit-llvm -o - -triple x86_64-apple-darwin10 %s | FileCheck %s

// Ensure that available_externally functions eliminated at -O2 are now
// declarations, and are not emitted as hidden with -fvisibility=hidden,
// but rather with default visibility.
struct Filter {
  virtual void Foo();
  int a;
};

class Message{};
class Sender {
 public:
  virtual bool Send(Message* msg) = 0;

 protected:
  virtual ~Sender() {}
};

// CHECK: declare noundef zeroext i1 @_ZThn16_N17SyncMessageFilter4SendEP7Message
class SyncMessageFilter : public Filter, public Sender {
 public:
  bool Send(Message* message) override;
};

class TestSyncMessageFilter : public SyncMessageFilter {
};

int main() {
  TestSyncMessageFilter *f = new TestSyncMessageFilter;
  f->Send(new Message);
}
