// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --builder 'clang' --debugger 'lldb' \
// RUN:     --cflags "-g -O2" -v -- %s
// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --builder 'clang' --debugger 'lldb' \
// RUN:     --cflags "-g -O0" -- %s

// REQUIRES: lldb
// UNSUPPORTED: system-windows

//// Check that the debugging experience with __attribute__((optnone)) at O2
//// matches O0. Test simple structs and methods.

long a_global_ptr[] = { 0xCAFEBABEL, 0xFEEDBEEFL };

namespace {

struct A {
  int a;
  float b;

  enum B {
    A_VALUE = 0x1,
    B_VALUE = 0x2
  };

  struct some_data {
    enum B other_b;
    enum B other_other_b;
  };

  struct other_data {
    union {
      void *raw_ptr;
      long  *long_ptr;
      float *float_ptr;
    } a;
    struct some_data b;
    struct some_data c;
  };
private:
  struct other_data _data;

public:
  struct other_data *getOtherData() { return &_data; }

  __attribute__((always_inline,nodebug))
  void setSomeData1(A::B value, A::B other_value) {
    struct other_data *data = getOtherData();
    data->b.other_b = value;
    data->b.other_other_b = other_value;
  }

  __attribute__((always_inline))
  void setSomeData2(A::B value, A::B other_value) {
    struct other_data *data = getOtherData();
    data->c.other_b = value;
    data->c.other_other_b = other_value;
  }

  void setOtherData() {
    setSomeData2(A_VALUE, B_VALUE);
    getOtherData()->a.long_ptr = &a_global_ptr[0];
  }

  __attribute__((optnone))
  A() {
    __builtin_memset(this, 0xFF, sizeof(*this));
  } //DexLabel('break_0')
  // DexExpectWatchValue('a', '-1', on_line=ref('break_0'))
  //// Check b is NaN by comparing it to itself.
  // DexExpectWatchValue('this->b == this->b', 'false', on_line=ref('break_0'))
  // DexExpectWatchValue('_data.a.raw_ptr == -1', 'true', on_line=ref('break_0'))
  // DexExpectWatchValue('_data.a.float_ptr == -1', 'true', on_line=ref('break_0'))
  // DexExpectWatchValue('_data.a.float_ptr == -1', 'true', on_line=ref('break_0'))
  // DexExpectWatchValue('a_global_ptr[0]', 0xcafebabe, on_line=ref('break_0'))
  // DexExpectWatchValue('a_global_ptr[1]', 0xfeedbeef, on_line=ref('break_0'))

  __attribute__((optnone))
  ~A() {
    *getOtherData()->a.long_ptr = 0xADDF00DL;
  } //DexLabel('break_1')
  // DexExpectWatchValue('_data.a.raw_ptr == a_global_ptr', 'true', on_line=ref('break_1'))
  // DexExpectWatchValue('a_global_ptr[0]', 0xaddf00d, on_line=ref('break_1'))

  __attribute__((optnone))
  long getData() {
    setSomeData1(B_VALUE, A_VALUE);
    setOtherData();
    return getOtherData()->a.long_ptr[1]; //DexLabel('break_2')
  }
  // DexExpectWatchValue('_data.b.other_b', 'B_VALUE', on_line=ref('break_2'))
  // DexExpectWatchValue('_data.b.other_other_b', 'A_VALUE', on_line=ref('break_2'))
};

} // anonymous namespace

int main() {
  int result = 0;
  {
    A a;
    result = a.getData();
  }
  return result;
}
