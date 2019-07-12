# Building libclang-cpp.so fails if LLVM_ENABLE_PIC=Off
if (NOT LLVM_ENABLE_PIC)
  return()
endif()

get_property(clang_libs GLOBAL PROPERTY CLANG_STATIC_LIBS)

foreach (lib ${clang_libs})
  list(APPEND _OBJECTS $<TARGET_OBJECTS:obj.${lib}>)
  # Use the static library for its dependencies. The objects that constitute the
  # static library will appear on the link line before the library, so it'll
  # just be ignored, but the dependencies of the library will still be linked
  # correctly.
  #
  # We could propagate the dependencies manually using the library's
  # INTERFACE_LINK_LIBRARIES property, but that will contain $<LINK_ONLY:...>
  # generator expressions if a static library has a private dependency, so we
  # can't use a $<TARGET_PROPERTY:...> generator expression to get the property
  # (since it wouldn't evaluate any generator expressions inside the property).
  # We could use get_property and do string manipulation to manually evaluate
  # the $<LINK_ONLY:...>, but that would miss any dependencies added after we
  # evaluate the get_property. We could also use the LINK_LIBRARIES property
  # instead, which should be free of any generator expressions, but that's
  # technically incorrect (it'd most likely work fine in practice, but still).
  #
  # Another alternative would be to use --whole-archive or equivalent instead of
  # using the object library at all. However, CMake reorders static libraries on
  # the link line so that a library appears after all its dependents, which can
  # reorder static libraries out of their --whole-archive --no-whole-archive
  # sandwich. It's really hard to avoid that reordering while still propagating
  # dependencies, which defeats the whole point.
  #
  # The ideal solution here is to bump our minimum CMake requirement to 3.12,
  # which adds support for dependencies on object libraries. Until then, linking
  # both the object and the static libraries seems like the least bad solution.
  #
  # TODO: Rework this when we bump our minimum CMake version to 3.12 or newer.
  list(APPEND _DEPS ${lib})
endforeach ()

add_clang_library(clang-cpp
                  SHARED
                  clang-shlib.cpp
                  ${_OBJECTS}
                  LINK_LIBS
                  ${_DEPS})
