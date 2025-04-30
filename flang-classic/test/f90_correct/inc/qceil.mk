#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qceil  ########


qceil: run
	

build:  $(SRC)/qceil.f08
	-$(RM) qceil.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qceil.f08 -o qceil.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qceil.$(OBJX) check.$(OBJX) $(LIBS) -o qceil.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qceil 
	qceil.$(EXESUFFIX)

verify: ;


