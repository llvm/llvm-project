#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test poly_ptr_to_str_1  ########


poly_ptr_to_str_1: run


build:  $(SRC)/poly_ptr_to_str_1.f90
	-$(RM) poly_ptr_to_str_1.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/poly_ptr_to_str_1.f90 -o poly_ptr_to_str_1.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) poly_ptr_to_str_1.$(OBJX) -o poly_ptr_to_str_1.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test poly_ptr_to_str_1
	poly_ptr_to_str_1.$(EXESUFFIX)

verify: ;

poly_ptr_to_str_1.run: run

