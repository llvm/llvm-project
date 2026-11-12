// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm -o - | FileCheck %s

// PR9322

// CHECK: @test1
// CHECK-NOT: switch
// CHECK-NOT: @dead
// CHECK: add nsw i32 {{.*}}, 1
// CHECK-NOT: switch
// CHECK-NOT: @dead
// CHECK: ret void
int i;
void dead(void);

void test1(void) {
  switch (1)
    case 1:
      ++i;

  switch (0)
    case 1:
      dead();
} 


// CHECK: @test2
// CHECK-NOT: switch
// CHECK-NOT: @dead
// CHECK: add nsw i32 {{.*}}, 2
// CHECK-NOT: switch
// CHECK-NOT: @dead
// CHECK: ret void
void test2(void) {
  switch (4) {
  case 1:
    dead();
    break;
  case 4:
    i += 2;
    // Fall off the end of the switch.
  } 
}


// CHECK: @test3
// CHECK-NOT: switch
// CHECK-NOT: @dead
// CHECK: add nsw i32 {{.*}}, 2
// CHECK-NOT: switch
// CHECK-NOT: @dead
// CHECK: ret void
void test3(void) {
  switch (4) {
  case 1:
    dead();
    break;
  case 4: {
    i += 2;
    break;
  }
  } 
}

// CHECK: @test4
// CHECK-NOT: switch
// CHECK-NOT: @dead
// CHECK: add nsw i32 {{.*}}, 2
// CHECK-NOT: switch
// CHECK-NOT: @dead
// CHECK: ret void
void test4(void) {
  switch (4) {
    case 1:
      dead();
      break;
    default: {
      i += 2;
      break;
    }
  } 
}

// This shouldn't crash codegen, but we don't have to optimize out the switch
// in this case.
void test5(void) {
  switch (1) {
    int x;  // eliding var decl?
    case 1:
      x = 4;
      i = x;
      break;
  } 
}

// CHECK: @test6
// CHECK-NOT: switch
// CHECK-NOT: @dead
// CHECK: ret void
void test6(void) {
  // Neither case is reachable.
  switch (40) {
  case 1:
   dead();
    break;
  case 4: {
    dead();
    break;
  }
  } 
}

// CHECK: @test7
// CHECK-NOT: switch
// CHECK-NOT: @dead
// CHECK: add nsw i32
// CHECK-NOT: switch
// CHECK-NOT: @dead
// CHECK: ret void
void test7(void) {
  switch (4) {
  case 1:
      dead();
    break;
    {
      case 4:   // crazy brace scenario
        ++i;
    }
    break;
  } 
}

// CHECK: @test8
// CHECK-NOT: switch
// CHECK-NOT: @dead
// CHECK: add nsw i32
// CHECK-NOT: switch
// CHECK-NOT: @dead
// CHECK: ret void
void test8(void) {
  switch (4) {
  case 1:
    dead();
    break;
  case 4:
    ++i;
    // Fall off the end of the switch.
  } 
}

// CHECK: @test9
// CHECK-NOT: switch
// CHECK-NOT: @dead
// CHECK: add nsw i32
// CHECK: add nsw i32
// CHECK-NOT: switch
// CHECK-NOT: @dead
// CHECK: ret void
void test9(int i) {
  switch (1) {
  case 5:
    dead();
  case 1:
    ++i;
    // Fall through is fine.
  case 4:
    ++i;
    break;
  } 
}

// CHECK: @test10
// CHECK-NOT: switch
// CHECK: ret i32
int test10(void) {
	switch(8) {
		case 8:
			break;
		case 4:
			break;
		default:
			dead();
	}
	
	return 0;
}

// CHECK: @test11
// CHECK-NOT: switch
// CHECK: ret void
void test11(void) {
  switch (1) {
    case 1:
      break;
    case 42: ;
      int x;  // eliding var decl?
      x = 4;
      break;
  }
}

// CHECK: @test12
// CHECK-NOT: switch
// CHECK: ret void
void test12(void) {
  switch (1) {
  case 2: {
     int a;   // Ok to skip this vardecl.
     a = 42;
   }
  case 1:
    break;
  case 42: ;
    int x;  // eliding var decl?
    x = 4;
    break;
  }
}

// Verify that case 42 only calls test14 once.
// CHECK: @test13
// CHECK: call void @test13(i32 noundef 97)
// CHECK-NEXT: br label %[[EPILOG2:[0-9.a-z]+]]
// CHECK: [[EPILOG2]]
// CHECK-NEXT: br label [[EPILOG:%[0-9.a-z]+]]
// CHECK: call void @test13(i32 noundef 42)
// CHECK-NEXT: br label [[EPILOG]]
void test13(int x) {
  switch (x) {
    case 42: test13(97);  // fallthrough
    case 11: break;
    default: test13(42); break;
  }
}

