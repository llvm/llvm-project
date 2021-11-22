// expected-warning@+1{{macro 'SYSTEM_MACRO' has been marked as final and should not be undefined}}
#undef SYSTEM_MACRO
// expected-warning@+1{{macro 'SYSTEM_MACRO' has been marked as final and should not be redefined}}
#define SYSTEM_MACRO WoahMoar
