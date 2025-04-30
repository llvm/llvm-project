#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop410  ########


oop410: run
	

build:  $(SRC)/oop410.f90
	-$(RM) oop410.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop410.f90 -o oop410.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop410.$(OBJX) check.$(OBJX) $(LIBS) -o oop410.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop410
	oop410.$(EXESUFFIX)

verify: ;

