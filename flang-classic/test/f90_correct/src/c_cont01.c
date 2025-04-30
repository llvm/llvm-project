/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

int pass_contiguous_array_c(const void *data, int m, int n, int *res) {
   const int *data_i = (const int *)data;
   for(int i = 0; i < m; i++) {
     for(int j = 0; j < n; j++) {
            res[i * n + j ] = data_i[i * n + j];
     }
   }
   return 0;
}
