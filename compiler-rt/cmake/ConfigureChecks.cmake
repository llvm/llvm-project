INCLUDE( CheckIncludeFile )
INCLUDE( CheckFunctionExists )
INCLUDE( CheckSymbolExists )
INCLUDE( CheckCSourceCompiles )

SET( PACKAGE ${PACKAGE_NAME} )
SET( VERSION ${PACKAGE_VERSION} )

SET( BINARYDIR ${CMAKE_BINARY_DIR} )
SET( SOURCEDIR ${CMAKE_SOURCE_DIR} )

# HEADER FILES
CHECK_INCLUDE_FILE( sys/byteorder.h HAVE_SYS_BYTEORDER_H )
CHECK_INCLUDE_FILE( AvailabilityMacros.h HAVE_AVAILABILITY_MACROS_H )
CHECK_INCLUDE_FILE( TargetConditionals.h HAVE_TARGET_CONDITIONALS_H )
CHECK_INCLUDE_FILE( libkern/OSAtomic.h HAVE_LIBKERN_OSATOMIC_H )

# FUNCTIONS
CHECK_FUNCTION_EXISTS( sysconf HAVE_SYSCONF )
CHECK_SYMBOL_EXISTS( OSAtomicCompareAndSwapInt libkern/OSAtomic.h HAVE_OSATOMIC_COMPARE_AND_SWAP_INT )
CHECK_SYMBOL_EXISTS( OSAtomicCompareAndSwapLong libkern/OSAtomic.h HAVE_OSATOMIC_COMPARE_AND_SWAP_LONG )

# BUILTIN
CHECK_C_SOURCE_COMPILES( "
volatile int a;
int main(int argc, char *argv[]) {
  (void)__sync_bool_compare_and_swap(&a, 1, 2);
  return 0;
}
" HAVE_SYNC_BOOL_COMPARE_AND_SWAP_INT )

CHECK_C_SOURCE_COMPILES( "
volatile long a;
int main(int argc, char *argv[]) {
  (void)__sync_bool_compare_and_swap(&a, 1, 2);
  return 0;
}
" HAVE_SYNC_BOOL_COMPARE_AND_SWAP_LONG )
