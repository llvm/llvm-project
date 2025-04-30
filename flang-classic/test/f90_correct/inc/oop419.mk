#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop419  ########


oop419: run
	

build:  $(SRC)/oop419.f90
	-$(RM) oop419.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop419.f90 -o oop419.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop419.$(OBJX) check.$(OBJX) $(LIBS) -o oop419.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop419
	oop419.$(EXESUFFIX)

verify: ;

