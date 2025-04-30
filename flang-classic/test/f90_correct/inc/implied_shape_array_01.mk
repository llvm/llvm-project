#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test implied_shape_array_01  ########


implied_shape_array_01: run


build:  $(SRC)/implied_shape_array_01.f08
	-$(RM) implied_shape_array_01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/implied_shape_array_01.f08 -o implied_shape_array_01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) implied_shape_array_01.$(OBJX) check.$(OBJX) $(LIBS) -o implied_shape_array_01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test implied_shape_array_01
	implied_shape_array_01.$(EXESUFFIX)

verify: ;
