#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test kqnint  ########


qp141: run
	

build:  $(SRC)/qp141.f08
	-$(RM) qp141.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o check_mod.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp141.f08 -o qp141.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp141.$(OBJX) check_mod.$(OBJX) $(LIBS) -o qp141.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qp141 
	qp141.$(EXESUFFIX)

verify: ;


