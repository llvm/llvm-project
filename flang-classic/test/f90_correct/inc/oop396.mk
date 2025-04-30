#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop396  ########


oop396: run
	

build:  $(SRC)/oop396.f90
	-$(RM) oop396.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop396.f90 -o oop396.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop396.$(OBJX) check.$(OBJX) $(LIBS) -o oop396.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop396
	oop396.$(EXESUFFIX)

verify: ;

