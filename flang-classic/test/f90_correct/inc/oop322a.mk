#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop322a  ########


oop322a: run
	

build:  $(SRC)/oop322a.f90
	-$(RM) oop322a.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop322a.f90 -o oop322a.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop322a.$(OBJX) check.$(OBJX) $(LIBS) -o oop322a.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop322a
	oop322a.$(EXESUFFIX)

verify: ;

