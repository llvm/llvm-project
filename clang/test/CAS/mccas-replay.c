// REQUIRES: x86-registered-target
// RUN: rm -rf %t && mkdir %t

// RUN: %clang -cc1depscan -o %t/args.rsp  -cc1-args -cc1 -triple x86_64-apple-darwin10 \
// RUN:    -debug-info-kind=standalone -dwarf-version=4 -debugger-tuning=lldb \
// RUN:    -emit-obj -fcas-backend  -fcas-path %t/cas       %s -o - > /dev/null

// RUN: %clang @%t/args.rsp -o %t/output1.o -Rcompile-job-cache 2> %t/output1.txt

// RUN: cat %t/output1.txt | grep llvmcas | sed \
// RUN:       -e "s/^.*miss for '//" \
// RUN:       -e "s/' .*$//" > %t/cache-key 

// RUN: c-index-test core -replay-cached-job -cas-path %t/cas @%t/cache-key \
// RUN:       -working-dir %t  -- @%t/args.rsp -o %t/output2.o

// RUN: diff %t/output1.o %t/output2.o 

int foo(int x) {
    return x+1;
}
