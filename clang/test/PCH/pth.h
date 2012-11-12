// This case came up when using PTH with Boost (<rdar://problem/8227989>).

# ifndef R8227989_PREPROCESSOR_CONFIG_CONFIG_HPP
# ifndef R8227989_PP_CONFIG_FLAGS
# endif
#
# ifndef R8227989_PP_CONFIG_ERRORS
#    ifdef NDEBUG
#    endif
# endif
# endif

