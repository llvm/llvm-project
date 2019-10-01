//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CONFIG C++ GC RR open rdar://6347910



struct MyStruct {
    int something;
};

struct TestObject {

        void test(void){
            {
                MyStruct first;   // works
            }
            void (^b)(void) = ^{ 
                MyStruct inner;  // fails to compile!
            };
        }
};

    

int main(int argc, char *argv[]) {
    printf("%s: Success\n", argv[0]);
    return 0;
}
