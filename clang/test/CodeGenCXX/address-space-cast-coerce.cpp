// RUN: %clang_cc1 %s -triple=amdgcn-amd-amdhsa -emit-llvm -o - | FileCheck %s

template<typename T, unsigned int n> struct my_vector_base;

    template<typename T>
    struct my_vector_base<T, 1> {
        typedef T Native_vec_ __attribute__((ext_vector_type(1)));

        union {
            Native_vec_ data;
            struct {
                T x;
            };
        };
    };

    template<typename T, unsigned int rank>
    struct my_vector_type : public my_vector_base<T, rank> {
        using my_vector_base<T, rank>::data;
        using typename my_vector_base<T, rank>::Native_vec_;

        template< typename U>
        my_vector_type(U x) noexcept
        {
            for (auto i = 0u; i != rank; ++i) data[i] = x;
        }
        my_vector_type& operator+=(const my_vector_type& x) noexcept
        {
            data += x.data;
            return *this;
        }
    };

template<typename T, unsigned int n>
    inline
    my_vector_type<T, n> operator+(
        const my_vector_type<T, n>& x, const my_vector_type<T, n>& y) noexcept
    {
        return my_vector_type<T, n>{x} += y;
    }

using char1 = my_vector_type<char, 1>;

int mane() {

    char1 f1{1};
    char1 f2{1};

// CHECK: [[CALL:%.*]] = call i16
// CHECK: [[TRUNC:%.*]] = trunc i16 [[CALL]] to i8
// CHECK: store i8 [[TRUNC]]

    char1 f3 = f1 + f2;
}
