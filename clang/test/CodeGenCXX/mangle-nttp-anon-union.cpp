// RUN: %clang_cc1 -std=c++20 -emit-llvm %s -o - -triple=x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -std=c++20 -emit-llvm %s -o - -triple=x86_64-linux-gnu | llvm-cxxfilt -n | FileCheck %s --check-prefix DEMANGLED

template<typename T>
struct wrapper1 {
  union {
    struct {
      T RightName;
    };
  };
};

template<typename T>
struct wrapper2 {
  union {
    struct {
      T RightName;
    };
    T WrongName;
  };
};

struct Base {
  int WrongName;
};

template <typename T>
struct wrapper3 {
  union {
    struct : Base {
      T RightName; };
    T WrongName;
  };
};

template <typename T>
struct wrapper4 {
  union {
    int RightName;
    struct {
      T WrongName;
    };
    T AlsoWrongName;
  };
};

template <typename T>
struct wrapper5 {
  union {
    struct {
      struct {
        T RightName;
      };
      T WrongName;
    };
  };
};

template<typename T>
struct wrapper6 {
  union {
    union{
    int : 5;
    T RightName;
    };
  };
};



template<auto tparam> void dummy(){}


void uses() {
  // Zero init'ed cases.
  dummy<wrapper1<int>{}>();
  // CHECK: call void @_Z5dummyIXtl8wrapper1IiEEEEvv
  // DEMANGLED: call void @void dummy<wrapper1<int>{}>()()
  dummy<wrapper2<float>{}>();
  // CHECK: call void @_Z5dummyIXtl8wrapper2IfEEEEvv
  // DEMANGLED: call void @void dummy<wrapper2<float>{}>()()
  dummy<wrapper3<short>{}>();
  // CHECK: call void @_Z5dummyIXtl8wrapper3IsEEEEvv
  // DEMANGLED: call void @void dummy<wrapper3<short>{}>()()
  dummy<wrapper4<double>{}>();
  // CHECK: call void @_Z5dummyIXtl8wrapper4IdEEEEvv
  // DEMANGLED: call void @void dummy<wrapper4<double>{}>()()
  dummy<wrapper5<long long>{}>();
  // CHECK: call void @_Z5dummyIXtl8wrapper5IxEEEEvv
  // DEMANGLED: call void @void dummy<wrapper5<long long>{}>()()
  dummy<wrapper6<int>{}>();
  // CHECK: call void @_Z5dummyIXtl8wrapper6IiEEEEvv
  // DEMANGLED: call void @void dummy<wrapper6<int>{}>()()

  dummy<wrapper1<double>{123.0}>();
  // CHECK: call void @_Z5dummyIXtl8wrapper1IdEtlNS1_Ut_Edi9RightNametlNS2_Ut_ELd405ec00000000000EEEEEEvv
  // DEMANGLED: call void @void dummy<wrapper1<double>{wrapper1<double>::'unnamed'{.RightName = wrapper1<double>::'unnamed'::'unnamed'{0x1.ec{{.*}}p+6}}}>()()
  dummy<wrapper2<double>{123.0}>();
  // CHECK: call void @_Z5dummyIXtl8wrapper2IdEtlNS1_Ut_Edi9RightNametlNS2_Ut_ELd405ec00000000000EEEEEEvv
  // DEMANGLED: call void @void dummy<wrapper2<double>{wrapper2<double>::'unnamed'{.RightName = wrapper2<double>::'unnamed'::'unnamed'{0x1.ec{{.*}}p+6}}}>()()
  dummy<wrapper3<double>{123, 456}>();
  // CHECK: call void @_Z5dummyIXtl8wrapper3IdEtlNS1_Ut_Edi9RightNametlNS2_Ut_Etl4BaseLi123EELd407c800000000000EEEEEEvv
  // DEMANGLED: call void @void dummy<wrapper3<double>{wrapper3<double>::'unnamed'{.RightName = wrapper3<double>::'unnamed'::'unnamed'{Base{123}, 0x1.c8{{.*}}p+8}}}>()()
  dummy<wrapper4<double>{123}>();
  // CHECK: call void @_Z5dummyIXtl8wrapper4IdEtlNS1_Ut_Edi9RightNameLi123EEEEEvv
  // DEMANGLED: call void @void dummy<wrapper4<double>{wrapper4<double>::'unnamed'{.RightName = 123}}>()()
  dummy<wrapper5<double>{123.0, 456.0}>();
  // CHECK: call void @_Z5dummyIXtl8wrapper5IdEtlNS1_Ut_Edi9RightNametlNS2_Ut_EtlNS3_Ut_ELd405ec00000000000EELd407c800000000000EEEEEEvv
  // DEMANGLED: call void @void dummy<wrapper5<double>{wrapper5<double>::'unnamed'{.RightName = wrapper5<double>::'unnamed'::'unnamed'{wrapper5<double>::'unnamed'::'unnamed'::'unnamed'{0x1.ec{{.*}}p+6}, 0x1.c8{{.*}}p+8}}}>()()
  dummy<wrapper6<double>{1}>();
  // CHECK: call void @_Z5dummyIXtl8wrapper6IdEtlNS1_Ut_Edi9RightNametlNS2_Ut_Edi9RightNameLd3ff0000000000000EEEEEEvv
  // DEMANGELD: call void @void dummy<wrapper6<double>{wrapper6<double>::'unnamed'{.RightName = wrapper6<double>::'unnamed'::'unnamed'{.RightName = 0x1{{.*}}p+0}}}>()()   
}
