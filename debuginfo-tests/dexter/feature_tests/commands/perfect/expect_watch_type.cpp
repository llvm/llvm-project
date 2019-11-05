// Purpose:
//      Check that \DexExpectWatchType applies no penalties when expected
//      types are found.
//
// REQUIRES: system-linux, lldb
//
// RUN: %dexter_base test --fail-lt 1.0 -w \
// RUN:     --builder 'clang' --debugger 'lldb' --cflags "-O0 -g" -- %s \
// RUN:     | FileCheck %s
// CHECK: expect_watch_type.cpp:

template<class T>
class Doubled {
public:
  Doubled(const T & to_double)
    : m_member(to_double * 2) {}

  T GetVal() {
    T to_return = m_member; // DexLabel('gv_start')
    return to_return;       // DexLabel('gv_end')
  }

  static T static_doubler(const T & to_double) {
    T result = 0;           // DexLabel('sd_start')
    result = to_double * 2;
    return result;          // DexLabel('sd_end')
  }

private:
  T m_member;
};

int main() {
  auto myInt = Doubled<int>(5); // DexLabel('main_start')
  auto myDouble = Doubled<double>(5.5);
  auto staticallyDoubledInt = Doubled<int>::static_doubler(5);
  auto staticallyDoubledDouble = Doubled<double>::static_doubler(5.5);
  return int(double(myInt.GetVal())
         + double(staticallyDoubledInt)
         + myDouble.GetVal()
         + staticallyDoubledDouble); // DexLabel('main_end')
}

// DexExpectWatchType('m_member', 'int', 'double', from_line='gv_start', to_line='gv_end')

// DexExpectWatchType('to_double', 'const int &', 'const double &', from_line='sd_start', to_line='sd_end')

// DexExpectWatchType('myInt', 'Doubled<int>', from_line='main_start', to_line='main_end')
// DexExpectWatchType('myDouble', 'Doubled<double>', from_line='main_start', to_line='main_end')
// DexExpectWatchType('staticallyDoubledInt', 'int', from_line='main_start', to_line='main_end')
// DexExpectWatchType('staticallyDoubledDouble', 'double', from_line='main_start', to_line='main_end')

