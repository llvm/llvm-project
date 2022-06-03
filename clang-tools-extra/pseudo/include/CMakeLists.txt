# The cxx.bnf grammar file
set(cxx_bnf ${CMAKE_CURRENT_SOURCE_DIR}/../lib/cxx.bnf)

set(CLANG_PSEUDO_GEN "clang-pseudo-gen" CACHE
  STRING "Host clang-pseudo-gen executable. Saves building if cross-compiling.")

if(NOT CLANG_PSEUDO_GEN STREQUAL "clang-pseudo-gen")
  set(pseudo_gen ${CLANG_PSEUDO_GEN})
  set(pseudo_gen_target ${CLANG_PSEUDO_GEN})
elseif(LLVM_USE_HOST_TOOLS)
  build_native_tool(clang-pseudo-gen pseudo_gen)
  set(pseudo_gen_target "${pseudo_gen}")
else()
  set(pseudo_gen $<TARGET_FILE:clang-pseudo-gen>)
  set(pseudo_gen_target clang-pseudo-gen)
endif()

# Generate inc files.
set(cxx_symbols_inc ${CMAKE_CURRENT_BINARY_DIR}/CXXSymbols.inc)
add_custom_command(OUTPUT ${cxx_symbols_inc}
   COMMAND "${pseudo_gen}"
     --grammar ${cxx_bnf}
     --emit-symbol-list
     -o ${cxx_symbols_inc}
   COMMENT "Generating nonterminal symbol file for cxx grammar..."
   DEPENDS ${pseudo_gen_target} ${cxx_bnf}
   VERBATIM)

set(cxx_bnf_inc ${CMAKE_CURRENT_BINARY_DIR}/CXXBNF.inc)
add_custom_command(OUTPUT ${cxx_bnf_inc}
   COMMAND "${pseudo_gen}"
     --grammar ${cxx_bnf}
     --emit-grammar-content
     -o ${cxx_bnf_inc}
   COMMENT "Generating bnf string file for cxx grammar..."
   DEPENDS ${pseudo_gen_target} ${cxx_bnf}
   VERBATIM)

# add_custom_command does not create a new target, we need to deine a target
# explicitly, so that other targets can depend on it.
add_custom_target(cxx_gen
    DEPENDS ${cxx_symbols_inc} ${cxx_bnf_inc}
    VERBATIM)
