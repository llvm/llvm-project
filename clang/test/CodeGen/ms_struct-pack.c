// RUN: %clang_cc1 -emit-llvm-only  -triple i386-apple-darwin9 -fdump-record-layouts %s | FileCheck %s

#pragma pack(1)
struct _one_ms {
        short m:9;      // size is 2
        int q:27;       // size is 6
        short w:13;     // size is 8
        short e:3;      // size is 8
        char r:4;       // size is 9
        char t:7;       // size is 10
        short y:16;     // size is 12
        short u:1;      // size is 14
        char i:2;       // size is 15
        int a;          // size is 19
        char o:6;       // size is 20
        char s:2;       // size is 20
        short d:10;     // size is 22
        short f:4;      // size is 22
        char b;         // size is 23
        char g:1;       // size is 24
        short h:13;     // size is 26
        char j:8;       // size is 27
        char k:5;       // size is 28
        char c;         // size is 29
        int l:28;       // size is 33
        char z:7;       // size is 34
        int x:20;       // size is 38
} __attribute__((__ms_struct__));
typedef struct _one_ms one_ms;

static int a1[(sizeof(one_ms) == 38) - 1];

#pragma pack(2)
struct _two_ms {
        short m:9;      
        int q:27;       
        short w:13;     
        short e:3;      
        char r:4;       
        char t:7;       
        short y:16;     
        short u:1;      
        char i:2;       
        int a;          
        char o:6;       
        char s:2;       
        short d:10;     
        short f:4;      
        char b;         
        char g:1;       
        short h:13;     
        char j:8;       
        char k:5;       
        char c;         
        int l:28;       
        char z:7;       
        int x:20;       
} __attribute__((__ms_struct__));

typedef struct _two_ms two_ms;

static int a2[(sizeof(two_ms) == 42) - 1];

#pragma pack(4)
struct _four_ms {
        short m:9;      
        int q:27;       
        short w:13;     
        short e:3;      
        char r:4;       
        char t:7;       
        short y:16;     
        short u:1;      
        char i:2;       
        int a;          
        char o:6;       
        char s:2;       
        short d:10;     
        short f:4;      
        char b;         
        char g:1;       
        short h:13;     
        char j:8;       
        char k:5;       
        char c;         
        int l:28;       
        char z:7;       
        int x:20;       
} __attribute__((__ms_struct__));
typedef struct _four_ms four_ms;

static int a4[(sizeof(four_ms) == 48) - 1];

#pragma pack(8)
struct _eight_ms {
        short m:9;      
        int q:27;       
        short w:13;     
        short e:3;      
        char r:4;       
        char t:7;       
        short y:16;     
        short u:1;      
        char i:2;       
        int a;          
        char o:6;       
        char s:2;       
        short d:10;     
        short f:4;      
        char b;         
        char g:1;       
        short h:13;     
        char j:8;       
        char k:5;       
        char c;         
        int l:28;       
        char z:7;       
        int x:20;       
} __attribute__((__ms_struct__));

typedef struct _eight_ms eight_ms;

static int a8[(sizeof(eight_ms) == 48) - 1];

#pragma pack(2)
struct test0 {
  unsigned long a : 8;
  unsigned long b : 8;
  unsigned long c : 8;
  unsigned long d : 10;
  unsigned long e : 1;
} __attribute__((__ms_struct__));

// CHECK:             0 | struct test0
// CHECK-NEXT:    0:0-7 |   unsigned long a
// CHECK-NEXT:    1:0-7 |   unsigned long b
// CHECK-NEXT:    2:0-7 |   unsigned long c
// CHECK-NEXT:    4:0-9 |   unsigned long d
// CHECK-NEXT:    5:2-2 |   unsigned long e
// CHECK-NEXT:          | [sizeof=8, align=2]

static int test0[(sizeof(struct test0) == 8) ? 1 : -1];
