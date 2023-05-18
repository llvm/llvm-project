// RUN: c-index-test core -print-source-symbols -- %s | FileCheck %s

// Functions taking function pointer parameters with different signatures should result in unique USRs.

typedef void (*_VoidToVoidPtr_)();
typedef void (*_IntToVoidPtr_)( int );
typedef _VoidToVoidPtr_ (*IntTo_VoidToVoidPtr_Ptr)( int );
typedef _IntToVoidPtr_ (*VoidTo_IntToVoidPtr_Ptr)();

void Func( IntTo_VoidToVoidPtr_Ptr );
// CHECK: {{[0-9]+}}:6 | function/C | Func | c:@F@Func#*F*Fv()(#I)# |
void Func( VoidTo_IntToVoidPtr_Ptr );
// CHECK: {{[0-9]+}}:6 | function/C | Func | c:@F@Func#*F*Fv(#I)()# |

void Func( void (* (*)(int, int))(int, int) );
// CHECK: {{[0-9]+}}:6 | function/C | Func | c:@F@Func#*F*Fv(#I#I)(#I#I)# |
void Func( void (* (*)(int, int, int))(int) );
// CHECK: {{[0-9]+}}:6 | function/C | Func | c:@F@Func#*F*Fv(#I)(#I#I#I)# |
