#ifndef SYCL_HPP
#define SYCL_HPP

// Shared code for SYCL tests

#ifndef __SYCL_DEVICE_ONLY__
#define __global
#endif

namespace cl {
namespace sycl {
namespace access {

enum class target {
  global_buffer = 2014,
  constant_buffer,
  local,
  image,
  host_buffer,
  host_image,
  image_array
};

enum class mode {
  read = 1024,
  write,
  read_write,
  discard_write,
  discard_read_write,
  atomic
};

enum class placeholder { false_t,
                         true_t };

enum class address_space : int {
  private_space = 0,
  global_space,
  constant_space,
  local_space
};
} // namespace access

template <int dim>
struct range {
};

template <int dim>
struct id {
};

template <int dim>
struct _ImplT {
    range<dim> AccessRange;
    range<dim> MemRange;
    id<dim> Offset;
};

template <typename dataT, int dimensions, access::mode accessmode,
          access::target accessTarget = access::target::global_buffer,
          access::placeholder isPlaceholder = access::placeholder::false_t>
class accessor {

public:
  void use(void) const {}
  void use(void*) const {}
  _ImplT<dimensions> impl;

private:
  void __init(__global dataT *Ptr, range<dimensions> AccessRange,
              range<dimensions> MemRange, id<dimensions> Offset) {}
};

struct sampler_impl {
#ifdef __SYCL_DEVICE_ONLY__
  __ocl_sampler_t m_Sampler;
#endif
};

class sampler {
  struct sampler_impl impl;
#ifdef __SYCL_DEVICE_ONLY__
  void __init(__ocl_sampler_t Sampler) { impl.m_Sampler = Sampler; }
#endif

public:
  void use(void) const {}
};

} // namespace sycl
} // namespace cl

#endif
