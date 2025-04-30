#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop467  ########


oop467: run
	

build:  $(SRC)/oop467.f90
	-$(RM) oop467.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop467.f90 -o oop467.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop467.$(OBJX) check.$(OBJX) $(LIBS) -o oop467.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop467
	oop467.$(EXESUFFIX)

verify: ;

