#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test io13  ########


io13: run
	

build:  $(SRC)/io13.f90
	-$(RM) io13.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/io13.f90 -o io13.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) io13.$(OBJX) check.$(OBJX) $(LIBS) -o io13.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test io13
	io13.$(EXESUFFIX)

verify: ;

