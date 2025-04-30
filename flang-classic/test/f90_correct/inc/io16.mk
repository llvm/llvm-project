#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test io16  ########


io16: run
	

build:  $(SRC)/io16.f90
	-$(RM) io16.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/io16.f90 -o io16.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) io16.$(OBJX) check.$(OBJX) $(LIBS) -o io16.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test io16
	io16.$(EXESUFFIX)

verify: ;

