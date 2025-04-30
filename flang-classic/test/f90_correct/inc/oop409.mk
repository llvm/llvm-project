#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop409  ########


oop409: run
	

build:  $(SRC)/oop409.f90
	-$(RM) oop409.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop409.f90 -o oop409.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop409.$(OBJX) check.$(OBJX) $(LIBS) -o oop409.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop409
	oop409.$(EXESUFFIX)

verify: ;

