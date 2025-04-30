#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka67  ########


ka67: run
	

build:  $(SRC)/ka67.f90
	-$(RM) ka67.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka67.f90 -o ka67.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka67.$(OBJX) check.$(OBJX) $(LIBS) -o ka67.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka67
	ka67.$(EXESUFFIX)

verify: ;

ka67.run: run

