!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s

!Ensure that there is no redundant LexicalBlock created with scope
!pointing to Subprogram and Local variable is pointing to Subprogram scope.
!CHECK: [[SCOPE_NODE:[0-9]+]] = distinct !DISubprogram(name: "sub", {{.*}}, line: [[LINE_NODE:[0-9]+]]
!CHECK: !DILocalVariable(name: "foo_arg", scope: ![[SCOPE_NODE]], file: !3, type: !8)
!CHECK-NOT: !DILexicalBlock(scope: ![[SCOPE_NODE]], {{.*}}, line: [[LINE_NODE]]

!Ensure that there is a LexicalBlock created for the BLOCK statement and
!the local variable `foo_block` has correct scope information i.e
!pointing to LexicalBlock.
!CHECK-DAG: !DILocalVariable(name: "foo_block", scope: ![[BLOCK_NODE:[0-9]+]]
!CHECK-DAG: ![[BLOCK_NODE]] = !DILexicalBlock(scope: ![[SCOPE_NODE]], {{.*}}, line: 19

SUBROUTINE sub(foo_arg)
      integer,value :: foo_arg
      integer :: foo_local
      foo_local = arg_foo
      BLOCK      !line number: 19
             integer :: foo_block
             foo_block = 4
      END BLOCK
END SUBROUTINE
