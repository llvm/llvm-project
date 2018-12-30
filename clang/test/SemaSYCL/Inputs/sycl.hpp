#ifndef SYCL_HPP
#define SYCL_HPP

// Shared code for SYCL tests

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
struct _ImplT {
    range<dim> Range;
};

template <typename dataT, int dimensions, access::mode accessmode,
          access::target accessTarget = access::target::global_buffer,
          access::placeholder isPlaceholder = access::placeholder::false_t>
class accessor {

public:
  void use(void) const {}
  void use(void*) const {}
  _ImplT<dimensions> __impl;
};

} // namespace sycl
} // namespace cl

#endif
