// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=debug.DumpDominators \
// RUN:   -analyzer-checker=debug.DumpPostDominators \
// RUN:   -analyzer-checker=debug.DumpControlDependencies \
// RUN:   2>&1 | FileCheck %s

// Test the DominatorsTree implementation with various control flows
int test1(void)
{
  int x = 6;
  int y = x/2;
  int z;

  while(y > 0) {
    if(y < x) {
      x = x/y;
      y = y-1;
    }else{
      z = x - y;
    }
    x = x - 1;
    x = x - 1;
  }
  z = x+y;
  z = 3;
  return 0;
}

// [B9 (ENTRY)] -> [B8] -> [B7] -> [B6] -> [B5] -> [B3] -> [B2]
//                          |\       \              /       /
//                          | \       ---> [B4] --->       /
//                          |  <---------------------------
//                          V
//                         [B1] -> [B0 (EXIT)]

// CHECK:      Control dependencies (Node#,Dependency#):
// CHECK-NEXT: (2,7)
// CHECK-NEXT: (3,7)
// CHECK-NEXT: (4,6)
// CHECK-NEXT: (4,7)
// CHECK-NEXT: (5,6)
// CHECK-NEXT: (5,7)
// CHECK-NEXT: (6,7)
// CHECK-NEXT: (7,7)
// CHECK-NEXT: Immediate dominance tree (Node#,IDom#):
// CHECK-NEXT: (0,1)
// CHECK-NEXT: (1,7)
// CHECK-NEXT: (2,3)
// CHECK-NEXT: (3,6)
// CHECK-NEXT: (4,6)
// CHECK-NEXT: (5,6)
// CHECK-NEXT: (6,7)
// CHECK-NEXT: (7,8)
// CHECK-NEXT: (8,9)
// CHECK-NEXT: (9,9)
// CHECK-NEXT: Immediate post dominance tree (Node#,IDom#):
// CHECK-NEXT: (0,0)
// CHECK-NEXT: (1,0)
// CHECK-NEXT: (2,7)
// CHECK-NEXT: (3,2)
// CHECK-NEXT: (4,3)
// CHECK-NEXT: (5,3)
// CHECK-NEXT: (6,3)
// CHECK-NEXT: (7,1)
// CHECK-NEXT: (8,7)
// CHECK-NEXT: (9,8)

int test2(void)
{
  int x,y,z;

  x = 10; y = 100;
  if(x > 0){
    y = 1;
  }else{
    while(x<=0){
      x++;
      y++;
    }
  }
  z = y;

  return 0;
}

//                                    <-------------
//                                   /              \
//                    -----------> [B4] -> [B3] -> [B2]
//                   /              |
//                  /               V
// [B7 (ENTRY)] -> [B6] -> [B5] -> [B1] -> [B0 (EXIT)]

// CHECK:      Control dependencies (Node#,Dependency#):
// CHECK-NEXT: (2,4)
// CHECK-NEXT: (2,6)
// CHECK-NEXT: (3,4)
// CHECK-NEXT: (3,6)
// CHECK-NEXT: (4,6)
// CHECK-NEXT: (4,4)
// CHECK-NEXT: (5,6)
// CHECK-NEXT: Immediate dominance tree (Node#,IDom#):
// CHECK-NEXT: (0,1)
// CHECK-NEXT: (1,6)
// CHECK-NEXT: (2,3)
// CHECK-NEXT: (3,4)
// CHECK-NEXT: (4,6)
// CHECK-NEXT: (5,6)
// CHECK-NEXT: (6,7)
// CHECK-NEXT: (7,7)
// CHECK-NEXT: Immediate post dominance tree (Node#,IDom#):
// CHECK-NEXT: (0,0)
// CHECK-NEXT: (1,0)
// CHECK-NEXT: (2,4)
// CHECK-NEXT: (3,2)
// CHECK-NEXT: (4,1)
// CHECK-NEXT: (5,1)
// CHECK-NEXT: (6,1)
// CHECK-NEXT: (7,6)

int test3(void)
{
  int x,y,z;

  x = y = z = 1;
  if(x>0) {
    while(x>=0){
      while(y>=x) {
        x = x-1;
        y = y/2;
      }
    }
  }
  z = y;

  return 0;
}

//                           <- [B2] <-
//                          /          \
// [B8 (ENTRY)] -> [B7] -> [B6] ---> [B5] -> [B4] -> [B3]
//                   \       |         \              /
//                    \      |          <-------------
//                     \      \
//                      --------> [B1] -> [B0 (EXIT)]

// CHECK:      Control dependencies (Node#,Dependency#):
// CHECK-NEXT: (2,6)
// CHECK-NEXT: (2,7)
// CHECK-NEXT: (3,5)
// CHECK-NEXT: (3,6)
// CHECK-NEXT: (3,7)
// CHECK-NEXT: (4,5)
// CHECK-NEXT: (4,6)
// CHECK-NEXT: (4,7)
// CHECK-NEXT: (5,6)
// CHECK-NEXT: (5,5)
// CHECK-NEXT: (5,7)
// CHECK-NEXT: (6,7)
// CHECK-NEXT: (6,6)
// CHECK-NEXT: Immediate dominance tree (Node#,IDom#):
// CHECK-NEXT: (0,1)
// CHECK-NEXT: (1,7)
// CHECK-NEXT: (2,5)
// CHECK-NEXT: (3,4)
// CHECK-NEXT: (4,5)
// CHECK-NEXT: (5,6)
// CHECK-NEXT: (6,7)
// CHECK-NEXT: (7,8)
// CHECK-NEXT: (8,8)
// CHECK-NEXT: Immediate post dominance tree (Node#,IDom#):
// CHECK-NEXT: (0,0)
// CHECK-NEXT: (1,0)
// CHECK-NEXT: (2,6)
// CHECK-NEXT: (3,5)
// CHECK-NEXT: (4,3)
// CHECK-NEXT: (5,2)
// CHECK-NEXT: (6,1)
// CHECK-NEXT: (7,1)
// CHECK-NEXT: (8,7)

int test4(void)
{
  int y = 3;
  while(y > 0) {
    if(y < 3) {
      while(y>0)
        y ++;
    }else{
      while(y<10)
        y ++;
    }
  }
  return 0;
}

//                               <----------------------------------
//                              /              <-----------------   \
//                             /              /                  \   \
// [B12 (ENTRY)] -> [B11] -> [B10]-> [B9] -> [B8] ---> [B7] -> [B6]  |
//                             |      \        \                     /
//                             |       \        -----> [B2] --------/
//                             |        \      /
//                             |          -> [B5] -> [B4] -> [B3]
//                             |               \              /
//                             |                <------------
//                              \
//                               -> [B1] -> [B0 (EXIT)]

// CHECK:      Control dependencies (Node#,Dependency#):
// CHECK-NEXT: (2,10)
// CHECK-NEXT: (3,5)
// CHECK-NEXT: (3,9)
// CHECK-NEXT: (3,10)
// CHECK-NEXT: (4,5)
// CHECK-NEXT: (4,9)
// CHECK-NEXT: (4,10)
// CHECK-NEXT: (5,9)
// CHECK-NEXT: (5,5)
// CHECK-NEXT: (5,10)
// CHECK-NEXT: (6,8)
// CHECK-NEXT: (6,9)
// CHECK-NEXT: (6,10)
// CHECK-NEXT: (7,8)
// CHECK-NEXT: (7,9)
// CHECK-NEXT: (7,10)
// CHECK-NEXT: (8,9)
// CHECK-NEXT: (8,8)
// CHECK-NEXT: (8,10)
// CHECK-NEXT: (9,10)
// CHECK-NEXT: (10,10)
// CHECK-NEXT: Immediate dominance tree (Node#,IDom#):
// CHECK-NEXT: (0,1)
// CHECK-NEXT: (1,10)
// CHECK-NEXT: (2,9)
// CHECK-NEXT: (3,4)
// CHECK-NEXT: (4,5)
// CHECK-NEXT: (5,9)
// CHECK-NEXT: (6,7)
// CHECK-NEXT: (7,8)
// CHECK-NEXT: (8,9)
// CHECK-NEXT: (9,10)
// CHECK-NEXT: (10,11)
// CHECK-NEXT: (11,12)
// CHECK-NEXT: (12,12)
// CHECK-NEXT: Immediate post dominance tree (Node#,IDom#):
// CHECK-NEXT: (0,0)
// CHECK-NEXT: (1,0)
// CHECK-NEXT: (2,10)
// CHECK-NEXT: (3,5)
// CHECK-NEXT: (4,3)
// CHECK-NEXT: (5,2)
// CHECK-NEXT: (6,8)
// CHECK-NEXT: (7,6)
// CHECK-NEXT: (8,2)
// CHECK-NEXT: (9,2)
// CHECK-NEXT: (10,1)
// CHECK-NEXT: (11,10)
// CHECK-NEXT: (12,11)

int test5(void)
{
  int x,y,z,a,b,c;
  x = 1;
  y = 2;
  z = 3;
  a = 4;
  b = 5;
  c = 6;
  if ( x < 10 ) {
     if ( y < 10 ) {
        if ( z < 10 ) {
           x = 4;
        } else {
           x = 5;
        }
        a = 10;
     } else {
       x = 6;
     }
     b = 10;
  } else {
    x = 7;
  }
  c = 11;
  return 0;
}

//                                                    [B0 (EXIT)] <--
//                                                                   \
// [B11 (ENTY)] -> [B10] -> [B9] -> [B8] -> [B7] -> [B5] -> [B3] -> [B1]
//                            |       |       |      /       /       /
//                            |       |       V     /       /       /
//                            |       V     [B6] -->       /       /
//                            V     [B4] ----------------->       /
//                          [B2]--------------------------------->

// CHECK:      Control dependencies (Node#,Dependency#):
// CHECK-NEXT: (2,10)
// CHECK-NEXT: (3,10)
// CHECK-NEXT: (4,9)
// CHECK-NEXT: (4,10)
// CHECK-NEXT: (5,9)
// CHECK-NEXT: (5,10)
// CHECK-NEXT: (6,8)
// CHECK-NEXT: (6,9)
// CHECK-NEXT: (6,10)
// CHECK-NEXT: (7,8)
// CHECK-NEXT: (7,9)
// CHECK-NEXT: (7,10)
// CHECK-NEXT: (8,9)
// CHECK-NEXT: (8,10)
// CHECK-NEXT: (9,10)
// CHECK-NEXT: Immediate dominance tree (Node#,IDom#):
// CHECK-NEXT: (0,1)
// CHECK-NEXT: (1,10)
// CHECK-NEXT: (2,10)
// CHECK-NEXT: (3,9)
// CHECK-NEXT: (4,9)
// CHECK-NEXT: (5,8)
// CHECK-NEXT: (6,8)
// CHECK-NEXT: (7,8)
// CHECK-NEXT: (8,9)
// CHECK-NEXT: (9,10)
// CHECK-NEXT: (10,11)
// CHECK-NEXT: (11,11)
// CHECK-NEXT: Immediate post dominance tree (Node#,IDom#):
// CHECK-NEXT: (0,0)
// CHECK-NEXT: (1,0)
// CHECK-NEXT: (2,1)
// CHECK-NEXT: (3,1)
// CHECK-NEXT: (4,3)
// CHECK-NEXT: (5,3)
// CHECK-NEXT: (6,5)
// CHECK-NEXT: (7,5)
// CHECK-NEXT: (8,5)
// CHECK-NEXT: (9,3)
// CHECK-NEXT: (10,1)
// CHECK-NEXT: (11,10)
