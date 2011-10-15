#
# Create a library called "CompilerRT" which includes the source files.

#INCLUDE_DIRECTORIES(
# ${CMAKE_CURRENT_BINARY_DIR}
#)

# Generic functions needed for each architecture
# libcompiler_rt.Generic.a    libcompiler_rt.Optimized.a

# Generic
SET( Generic_SRCS
 absvdi2.c absvsi2.c addvdi3.c addvsi3.c ashldi3.c ashrdi3.c
 clzdi2.c clzsi2.c cmpdi2.c ctzdi2.c ctzsi2.c
 divdc3.c divdi3.c divsc3.c ffsdi2.c
 fixdfdi.c fixsfdi.c fixunsdfdi.c fixunsdfsi.c fixunssfdi.c
 fixunssfsi.c floatdidf.c floatdisf.c floatundidf.c floatundisf.c
 gcc_personality_v0.c lshrdi3.c moddi3.c muldc3.c muldi3.c
 mulsc3.c mulvdi3.c mulvsi3.c negdi2.c negvdi2.c negvsi2.c
 paritydi2.c paritysi2.c popcountdi2.c popcountsi2.c powidf2.c
 powisf2.c subvdi3.c subvsi3.c ucmpdi2.c udivdi3.c
 udivmoddi4.c umoddi3.c apple_versioning.c eprintf.c
 )

# Optimized functions for each architecture

# Commenting out for the min until the basics are working first.
# ADD_SUBDIRECTORY( ppc )
# ADD_SUBDIRECTORY( x86_64 )
# ADD_SUBDIRECTORY( i386 )
# ADD_SUBDIRECTORY( arm )

# List of functions needed for each architecture.
SET( i386_Functions
 divxc3.c fixunsxfdi.c fixunsxfsi.c fixxfdi.c floatdixf.c
 floatundixf.c mulxc3.c powixf2.c clear_cache.c enable_execute_stack.c 
 )

SET( x86_64_Functions
 absvti2.c addvti3.c ashlti3.c ashrti3.c clzti2.c cmpti2.c
 ctzti2.c divti3.c divxc3.c ffsti2.c fixdfti.c fixsfti.c
 fixunsdfti.c fixunssfti.c fixunsxfdi.c fixunsxfsi.c
 fixunsxfti.c fixxfdi.c fixxfti.c floatdixf.c floattidf.c
 floattisf.c floattixf.c floatundixf.c floatuntidf.c
 floatuntisf.c floatuntixf.c lshrti3.c modti3.c multi3.c
 mulvti3.c mulxc3.c negti2.c negvti2.c parityti2.c
 popcountti2.c powixf2.c subvti3.c ucmpti2.c udivmodti4.c
 udivti3.c umodti3.c clear_cache.c enable_execute_stack.c
 )

SET( PPC_Functions
 divtc3.c fixtfdi.c fixunstfdi.c floatditf.c floatunditf.c
 gcc_qadd.c gcc_qdiv.c gcc_qmul.c gcc_qsub.c multc3.c
 powitf2.c restFP.c saveFP.c trampoline_setup.c
 clear_cache.c enable_execute_stack.c
 )

SET( ARM_Functions
 adddf3vfp.c addsf3vfp.c bswapdi2.c bswapsi2.c divdf3vfp.c
 divsf3vfp.c eqdf2vfp.c eqsf2vfp.c extendsfdf2vfp.c
 fixdfsivfp.c fixsfsivfp.c fixunsdfsivfp.c fixunssfsivfp.c
 floatsidfvfp.c floatsisfvfp.c floatunssidfvfp.c floatunssisfvfp.c
 gedf2vfp.c gesf2vfp.c gtdf2vfp.c gtsf2vfp.c
 ledf2vfp.c lesf2vfp.c ltdf2vfp.c ltsf2vfp.c
 muldf3vfp.c mulsf3vfp.c
 nedf2vfp.c negdf2vfp.c negsf2vfp.c nesf2vfp.c
 subdf3vfp.c subsf3vfp.c truncdfsf2vfp.c unorddf2vfp.c unordsf2vfp.c
 modsi3.c umodsi3.c udivsi3.c divsi3.c switch.c
 )

#FOREACH( LOOP_VAR ${Achitectures} )
# See ARCHIVE_OUTPUT_DIRECTORY docs.
#${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${LOOP_VAR}
#ENDFOREACH

ADD_LIBRARY( ${PROJECT_NAME}-Generic STATIC ${Generic_SRCS} )
#ADD_LIBRARY( ${PROJECT_NAME}-i386 STATIC ${i386_Functions} )

# [[debug|optimized|general]
#TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${PROJECT_NAME}-Common optimized ${PROJECT_NAME}-i386 )

