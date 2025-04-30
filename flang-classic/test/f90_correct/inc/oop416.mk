#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop416  ########


oop416: run
	

build:  $(SRC)/oop416.f90
	-$(RM) oop416.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop416.f90 -o oop416.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop416.$(OBJX) check.$(OBJX) $(LIBS) -o oop416.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop416
	oop416.$(EXESUFFIX)

verify: ;

