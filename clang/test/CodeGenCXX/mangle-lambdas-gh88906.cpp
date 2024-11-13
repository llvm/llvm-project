// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -emit-llvm -mconstructor-aliases -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fclang-abi-compat=18 %s -emit-llvm -mconstructor-aliases  -o - | FileCheck --check-prefix=CLANG18 %s
// RUN: %clang_cc1 -triple i386-pc-win32 %s -emit-llvm -mconstructor-aliases -o - | FileCheck --check-prefix=MSABI %s


class func {
public:
    template <typename T>
    func(T){};
    template <typename T, typename U>
    func(T, U){};
};

void GH88906(){
  class Test{
    public:
    func a{[]{ }, []{ }};
    func b{[]{ }};
    func c{[]{ }};
  } test;
}

// CHECK-LABEL: define internal void @_ZZ7GH88906vEN4TestC2Ev
// CHECK: call void @_ZN4funcC2IN7GH889064Test1aMUlvE_ENS3_UlvE0_EEET_T0_
// CHECK: call void @_ZN4funcC2IN7GH889064Test1bMUlvE_EEET_
// CHECK: call void @_ZN4funcC2IN7GH889064Test1cMUlvE_EEET_

// CHECK-LABEL: define internal void @_ZN4funcC2IN7GH889064Test1aMUlvE_ENS3_UlvE0_EEET_T0_
// CHECK-LABEL: define internal void @_ZN4funcC2IN7GH889064Test1bMUlvE_EEET_
// CHECK-LABEL: define internal void @_ZN4funcC2IN7GH889064Test1cMUlvE_EEET_

// CLANG18-LABEL: define internal void @_ZZ7GH88906vEN4TestC2Ev
// CLANG18: call void @_ZN4funcC2IZ7GH88906vEN4TestUlvE_EZ7GH88906vENS1_UlvE0_EEET_T0_
// CLANG18: call void @_ZN4funcC2IZ7GH88906vEN4TestUlvE_EEET_
// CLANG18: call void @_ZN4funcC2IZ7GH88906vEN4TestUlvE_EEET_



// MSABI-LABEL: define internal x86_thiscallcc noundef ptr @"??0Test@?1??GH88906@@YAXXZ@QAE@XZ"
// MSABI: call x86_thiscallcc noundef ptr @"??$?0V<lambda_1>@a@Test@?1??GH88906@@YAXXZ@V<lambda_2>@12?1??3@YAXXZ@@func@@QAE@V<lambda_1>@a@Test@?1??GH88906@@YAXXZ@V<lambda_2>@23?1??4@YAXXZ@@Z"
// MSABI: call x86_thiscallcc noundef ptr @"??$?0V<lambda_1>@b@Test@?1??GH88906@@YAXXZ@@func@@QAE@V<lambda_1>@b@Test@?1??GH88906@@YAXXZ@@Z"
// MSABI: call x86_thiscallcc noundef ptr @"??$?0V<lambda_1>@c@Test@?1??GH88906@@YAXXZ@@func@@QAE@V<lambda_1>@c@Test@?1??GH88906@@YAXXZ@@Z"

// MSABI-LABEL: define internal x86_thiscallcc noundef ptr @"??$?0V<lambda_1>@a@Test@?1??GH88906@@YAXXZ@V<lambda_2>@12?1??3@YAXXZ@@func@@QAE@V<lambda_1>@a@Test@?1??GH88906@@YAXXZ@V<lambda_2>@23?1??4@YAXXZ@@Z"
// MSABI-LABEL: define internal x86_thiscallcc noundef ptr @"??$?0V<lambda_1>@b@Test@?1??GH88906@@YAXXZ@@func@@QAE@V<lambda_1>@b@Test@?1??GH88906@@YAXXZ@@Z"
// MSABI-LABEL: define internal x86_thiscallcc noundef ptr @"??$?0V<lambda_1>@c@Test@?1??GH88906@@YAXXZ@@func@@QAE@V<lambda_1>@c@Test@?1??GH88906@@YAXXZ@@Z"
