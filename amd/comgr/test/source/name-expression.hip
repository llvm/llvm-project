template<int N, typename T>
__attribute__((global)) void my_kernel_FOO(T* array) {
    array[0] = N;
}
static __attribute__((device)) const void* __amdgcn_name_expr_ABC[] = {
   "my_kernel_FOO<static_cast<int>(2+1),float >",
    (void*)&my_kernel_FOO<static_cast<int>(2+1),float >
    };

static auto __amdgcn_name_expr_stub_ABC = __amdgcn_name_expr_ABC;

template<int N, typename T>
__attribute__((global)) void my_kernel_BOO(T* array) {
    array[0] = N;
}
static __attribute__((device)) const void* __amdgcn_name_expr_XYZ[] = {
   "my_kernel_BOO<static_cast<int>(2+1),float >",
    (void*)&my_kernel_BOO<static_cast<int>(2+1),float >
    };

static auto __amdgcn_name_expr_stub_XYZ= __amdgcn_name_expr_XYZ;
