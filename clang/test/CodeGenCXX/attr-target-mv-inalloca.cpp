// RUN: %clang_cc1 -std=c++11 -triple i686-windows-msvc -emit-llvm %s -o - | FileCheck %s --check-prefix=WINDOWS
// RUN: %clang_cc1 -std=c++11 -triple x86_64-windows-msvc -emit-llvm %s -o - | FileCheck %s --check-prefix=WINDOWS64

struct Foo {
  Foo();
  Foo(const Foo &o);
  ~Foo();
  int x;
};
int __attribute__((target("default"))) bar(Foo o) { return o.x; }
int __attribute__((target("sse4.2"))) bar(Foo o) { return o.x + 1; }
int __attribute__((target("arch=ivybridge"))) bar(Foo o) { return o.x + 2; }

void usage() {
  Foo f;
  bar(f);
}

// WINDOWS: define dso_local i32 @"?bar@@YAHUFoo@@@Z"(<{ %struct.Foo }>* inalloca(<{ %struct.Foo }>) %0)
// WINDOWS: %[[O:[0-9a-zA-Z]+]] = getelementptr inbounds <{ %struct.Foo }>, <{ %struct.Foo }>* %0, i32 0, i32 0
// WINDOWS: %[[X:[0-9a-zA-Z]+]] = getelementptr inbounds %struct.Foo, %struct.Foo* %[[O]], i32 0, i32 0
// WINDOWS: %[[LOAD:[0-9a-zA-Z]+]] = load i32, i32* %[[X]]
// WINDOWS: ret i32 %[[LOAD]]

// WINDOWS: define dso_local i32 @"?bar@@YAHUFoo@@@Z.sse4.2"(<{ %struct.Foo }>* inalloca(<{ %struct.Foo }>) %0)
// WINDOWS: %[[O:[0-9a-zA-Z]+]] = getelementptr inbounds <{ %struct.Foo }>, <{ %struct.Foo }>* %0, i32 0, i32 0
// WINDOWS: %[[X:[0-9a-zA-Z]+]] = getelementptr inbounds %struct.Foo, %struct.Foo* %[[O]], i32 0, i32 0
// WINDOWS: %[[LOAD:[0-9a-zA-Z]+]] = load i32, i32* %[[X]]
// WINDOWS: %[[ADD:[0-9a-zA-Z]+]] = add nsw i32 %[[LOAD]], 1
// WINDOWS: ret i32 %[[ADD]]

// WINDOWS: define dso_local i32 @"?bar@@YAHUFoo@@@Z.arch_ivybridge"(<{ %struct.Foo }>* inalloca(<{ %struct.Foo }>) %0)
// WINDOWS: %[[O:[0-9a-zA-Z]+]] = getelementptr inbounds <{ %struct.Foo }>, <{ %struct.Foo }>* %0, i32 0, i32 0
// WINDOWS: %[[X:[0-9a-zA-Z]+]] = getelementptr inbounds %struct.Foo, %struct.Foo* %[[O]], i32 0, i32 0
// WINDOWS: %[[LOAD:[0-9a-zA-Z]+]] = load i32, i32* %[[X]]
// WINDOWS: %[[ADD:[0-9a-zA-Z]+]] = add nsw i32 %[[LOAD]], 2
// WINDOWS: ret i32 %[[ADD]]

// WINDOWS: define dso_local void @"?usage@@YAXXZ"()
// WINDOWS: %[[F:[0-9a-zA-Z]+]] = alloca %struct.Foo
// WINDOWS: %[[ARGMEM:[0-9a-zA-Z]+]] = alloca inalloca <{ %struct.Foo }>
// WINDOWS: %[[CALL:[0-9a-zA-Z]+]] = call i32 @"?bar@@YAHUFoo@@@Z.resolver"(<{ %struct.Foo }>* inalloca(<{ %struct.Foo }>) %[[ARGMEM]])

// WINDOWS: define weak_odr dso_local i32 @"?bar@@YAHUFoo@@@Z.resolver"(<{ %struct.Foo }>* %0)
// WINDOWS: %[[RET:[0-9a-zA-Z]+]] = musttail call i32 @"?bar@@YAHUFoo@@@Z.arch_ivybridge"(<{ %struct.Foo }>* %0)
// WINDOWS-NEXT: ret i32 %[[RET]]
// WINDOWS: %[[RET:[0-9a-zA-Z]+]] = musttail call i32 @"?bar@@YAHUFoo@@@Z.sse4.2"(<{ %struct.Foo }>* %0)
// WINDOWS-NEXT: ret i32 %[[RET]]
// WINDOWS: %[[RET:[0-9a-zA-Z]+]] = musttail call i32 @"?bar@@YAHUFoo@@@Z"(<{ %struct.Foo }>* %0)
// WINDOWS-NEXT: ret i32 %[[RET]]


// WINDOWS64: define dso_local i32 @"?bar@@YAHUFoo@@@Z"(%struct.Foo* %[[O:[0-9a-zA-Z]+]])
// WINDOWS64: %[[X:[0-9a-zA-Z]+]] = getelementptr inbounds %struct.Foo, %struct.Foo* %[[O]], i32 0, i32 0
// WINDOWS64: %[[LOAD:[0-9a-zA-Z]+]] = load i32, i32* %[[X]]
// WINDOWS64: ret i32 %[[LOAD]]

// WINDOWS64: define dso_local i32 @"?bar@@YAHUFoo@@@Z.sse4.2"(%struct.Foo* %[[O:[0-9a-zA-Z]+]])
// WINDOWS64: %[[X:[0-9a-zA-Z]+]] = getelementptr inbounds %struct.Foo, %struct.Foo* %[[O]], i32 0, i32 0
// WINDOWS64: %[[LOAD:[0-9a-zA-Z]+]] = load i32, i32* %[[X]]
// WINDOWS64: %[[ADD:[0-9a-zA-Z]+]] = add nsw i32 %[[LOAD]], 1
// WINDOWS64: ret i32 %[[ADD]]

// WINDOWS64: define dso_local i32 @"?bar@@YAHUFoo@@@Z.arch_ivybridge"(%struct.Foo* %[[O:[0-9a-zA-Z]+]])
// WINDOWS64: %[[X:[0-9a-zA-Z]+]] = getelementptr inbounds %struct.Foo, %struct.Foo* %[[O]], i32 0, i32 0
// WINDOWS64: %[[LOAD:[0-9a-zA-Z]+]] = load i32, i32* %[[X]]
// WINDOWS64: %[[ADD:[0-9a-zA-Z]+]] = add nsw i32 %[[LOAD]], 2
// WINDOWS64: ret i32 %[[ADD]]

// WINDOWS64: define dso_local void @"?usage@@YAXXZ"()
// WINDOWS64: %[[F:[0-9a-zA-Z]+]] = alloca %struct.Foo
// WINDOWS64: %[[ARG:[0-9a-zA-Z.]+]] = alloca %struct.Foo
// WINDOWS64: %[[CALL:[0-9a-zA-Z]+]] = call i32 @"?bar@@YAHUFoo@@@Z.resolver"(%struct.Foo* %[[ARG]])

// WINDOWS64: define weak_odr dso_local i32 @"?bar@@YAHUFoo@@@Z.resolver"(%struct.Foo* %0)
// WINDOWS64: %[[RET:[0-9a-zA-Z]+]] = musttail call i32 @"?bar@@YAHUFoo@@@Z.arch_ivybridge"(%struct.Foo* %0)
// WINDOWS64-NEXT: ret i32 %[[RET]]
// WINDOWS64: %[[RET:[0-9a-zA-Z]+]] = musttail call i32 @"?bar@@YAHUFoo@@@Z.sse4.2"(%struct.Foo* %0)
// WINDOWS64-NEXT: ret i32 %[[RET]]
// WINDOWS64: %[[RET:[0-9a-zA-Z]+]] = musttail call i32 @"?bar@@YAHUFoo@@@Z"(%struct.Foo* %0)
// WINDOWS64-NEXT: ret i32 %[[RET]]
