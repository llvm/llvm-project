#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test io10  ########


io10: run
	

build:  $(SRC)/io10.f90
	-$(RM) io10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/io10.f90 -o io10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) io10.$(OBJX) check.$(OBJX) $(LIBS) -o io10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test io10
	io10.$(EXESUFFIX)

verify: ;

