#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test pack function take quadruple precision  ########


qpack: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qpack.f08 fcheck.$(OBJX)
	-$(RM) qpack.$(EXESUFFIX) qpack.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qpack.f08 -o qpack.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qpack.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qpack.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qpack
	qpack.$(EXESUFFIX)

verify: ;

qpack.run: run

