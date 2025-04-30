#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test poly_ptr_to_str_2  ########


poly_ptr_to_str_2: run


build:  $(SRC)/poly_ptr_to_str_2.f90
	-$(RM) poly_ptr_to_str_2.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/poly_ptr_to_str_2.f90 -o poly_ptr_to_str_2.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) poly_ptr_to_str_2.$(OBJX) -o poly_ptr_to_str_2.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test poly_ptr_to_str_2
	poly_ptr_to_str_2.$(EXESUFFIX)

verify: ;

poly_ptr_to_str_2.run: run

