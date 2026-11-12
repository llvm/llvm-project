! RUN: %flang -E %s 2>&1 | FileCheck %s
!CHECK: CALL myfunc( 'hello ' // 'world' // 'again')
#define NOCOMMENT
NOCOMMENT CALL myfunc( 'hello ' // &
NOCOMMENT 'world' // &
NOCOMMENT 'again' )
end
