#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
red03: red03.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN2) ./a.$(EXESUFFIX) $(LOG)
red03.$(OBJX): $(SRC)/red03.f90 check.$(OBJX)
	@echo ------------ building test $@
	-$(F90) $(FFLAGS) $(SRC)/red03.f90
	@$(RM) ./a.$(EXESUFFIX)
	-$(F90) $(LDFLAGS) red03.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: red03
run: ;
