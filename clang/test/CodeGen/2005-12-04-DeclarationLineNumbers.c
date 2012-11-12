// RUN: %clang_cc1 %s -emit-llvm -g -o - | grep DW_TAG_compile_unit | count 1
// PR664: ensure that line #'s are emitted for declarations


short test(short br_data_0,
short br_data_1,
short br_data_2,
short br_data_3,
short br_data_4,
short br_data_5,
short br_data_6,
short br_data_7) {

short sm07 = br_data_0 + br_data_7;
short sm16 = br_data_1 + br_data_6;
short sm25 = br_data_2 + br_data_5;
short sm34 = br_data_3 + br_data_4;
short s0734 = sm07 + sm34;
short s1625 = sm16 + sm25;

return s0734 + s1625;
}

