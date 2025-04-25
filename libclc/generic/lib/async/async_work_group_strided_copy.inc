
#define STRIDED_COPY(dst, src, num_gentypes, dst_stride, src_stride)       \
  size_t size = get_local_size(0) * get_local_size(1) * get_local_size(2); \
  size_t id = (get_local_size(1) * get_local_size(2) * get_local_id(0)) +  \
              (get_local_size(2) * get_local_id(1)) +                      \
              get_local_id(2);                                             \
  size_t i;                                                                \
                                                                           \
  for (i = id; i < num_gentypes; i += size) {                              \
    dst[i * dst_stride] = src[i * src_stride];                             \
  }


_CLC_OVERLOAD _CLC_DEF event_t async_work_group_strided_copy(
    local __CLC_GENTYPE *dst,
    const global __CLC_GENTYPE *src,
    size_t num_gentypes,
    size_t src_stride,
    event_t event) {

  STRIDED_COPY(dst, src, num_gentypes, 1, src_stride);
  return event;
}

_CLC_OVERLOAD _CLC_DEF event_t async_work_group_strided_copy(
    global __CLC_GENTYPE *dst,
    const local __CLC_GENTYPE *src,
    size_t num_gentypes,
    size_t dst_stride,
    event_t event) {

  STRIDED_COPY(dst, src, num_gentypes, dst_stride, 1);
  return event;
}
