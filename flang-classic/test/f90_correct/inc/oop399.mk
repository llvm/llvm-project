#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop399  ########


oop399: run
	

build:  $(SRC)/oop399.f90
	-$(RM) oop399.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop399.f90 -o oop399.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop399.$(OBJX) check.$(OBJX) $(LIBS) -o oop399.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop399
	oop399.$(EXESUFFIX)

verify: ;

