// RUN: %clang_cc1 -triple i686-w64-windows-gnu -o - -emit-llvm -debug-info-kind=constructor %s | FileCheck %s

enum nsresult {};

class NotNull;

class nsICanvasRenderingContextInternal {
  // CHECK: !DISubprogram(name: "InitializeWithDrawTarget", linkageName: "\01__ZN33nsICanvasRenderingContextInternal24InitializeWithDrawTargetE7NotNull@4"
  nsresult __stdcall InitializeWithDrawTarget(NotNull);
} nsTBaseHashSet;
