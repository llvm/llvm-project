__kernel void foo(int *i) {
  i[get_group_id(0)] = 1;
}
