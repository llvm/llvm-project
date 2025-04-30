/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief Nroff to Sphinx conversion utility.

   This program takes arbitrary number of input file names on the command line
   and creates corresponding files with the extension "rst", converting the
   input files from Nroff to Sphinx format.
 */

#include "utils.h"
#include <iostream>

/**
   \class is the main application class.
 */
class N2Rst : public UtilityApplication
{

  bool verbose; /// whether the program should print what it does.
  std::vector<std::string> input_filenames; /// the arrays of input filenames.

public:
  /**
     Constructor takes as argument the vector of command line options given to
     the program and assumes args doesn't contain the program name as args[0].
   */
  N2Rst(const std::vector<std::string> &args) : verbose(false)
  {
    for (std::vector<std::string>::const_iterator it = args.begin(),
                                                  E = args.end();
         it != E; ++it) {
      if (*it != "-v") {
        input_filenames.push_back(*it);
      } else {
        verbose = true;
      }
    }
  }

  /**
     \brief Implements the logic of the program.
   */
  int
  run()
  {
    if (input_filenames.empty()) {
      std::cerr << "No input files." << std::endl;
      return 1;
    }
    for (std::vector<std::string>::const_iterator it = input_filenames.begin(),
                                                  E = input_filenames.end();
         it != E; ++it) {
      std::ifstream ifs(it->c_str());
      if (!ifs) {
        std::cerr << "Can't open file " << *it << std::endl;
        return 1;
      }
      if (verbose) {
        std::cout << *it << " -> " << get_output_filename(*it) << std::endl;
      }
      sphinx.setOutputFile(get_output_filename(*it));
      for (std::string line; std::getline(ifs, line);) {
        sphinx.process(line);
      }
    }
    return 0;
  }

private:
  /**
     \brief Generate the name of an output file given the name of an input file.
   */
  std::string
  get_output_filename(const std::string &s)
  {
    auto pos = s.find_last_of('/');
    std::string filename = pos != std::string::npos ? s.substr(pos + 1) : s;
    pos = filename.find_last_of('.');
    if (pos != std::string::npos) {
      return filename.substr(0, pos + 1) + "rst";
    }
    return filename + ".rst";
  }
};

int
main(int argc, char **argv)
{
  // DO NOT add the program name to the vector of arguments.
  N2Rst app(std::vector<std::string>(argv + 1, argv + argc));
  return app.run();
}
