#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fe40  ########


fe40: run
	

build:  $(SRC)/fe40.f90
	-$(RM) fe40.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fe40.f90 -o fe40.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fe40.$(OBJX) check.$(OBJX) $(LIBS) -o fe40.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fe40
	fe40.$(EXESUFFIX)

verify: ;

fe40.run: run

