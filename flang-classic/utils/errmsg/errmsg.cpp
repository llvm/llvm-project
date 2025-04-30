/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>

/**
 * \file errmsg.cpp
 * \brief utility program for managing compiler error messages.
 *
 *  ERRMSG - Utility program which reads file(s) in nroff format
 *           defining error numbers and messages, and writes a file of
 *           C code defining and initializing the data structure
 *           containing the message text.
 *
 *  INPUT:   error message text definition file(s).
 *
 *  OUTPUT:  errmsgdf.h
 *           output format:
 *           static const char* errtxt[] = {
 *             ** 000 ** "text for message 0",
 *             ** 001 ** "text for message 1",
 *                       " . . . ",
 *           };
 *
 *  An error message definition in the errmsg*.n files looks like this:
 *
 *  .MS W 109 "Type specification of field $ ignored"
 *  Bit fields must be int, char, or short.
 *  Bit field is given the type unsigned int.
 *
 *  The arguments to the .MS macro are:
 *
 *  - A letter indicating the severity. One of:
 *    [I]nformational, [W]arning, [S]evere error,
 *    [F]atal error, or [V]ariable severity.
 *
 *  - A unique error code.
 *
 *  - The error message in quotes.
 *
 *  - Optionaly, a symbolic name for referring to the error in source
 *    code.  If no symbolic name is given, one is derived from the
 *    error message like this:
 *
 *    W_0109_Type_specification_of_field_OP1_ignored
 *
 *  When the symbolic name is specified explicitly, it replaces the
 *  automatically generated name.
 */

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

/**
 * \class Message
 * \brief Record for each message.
 */
struct Message {
  ssize_t number;                 //!< numeric error code
  char severity;                  //!< error severity code
  std::string symbol;             //!< enum symbol for the error
  std::string message;            //!< error message text
  std::vector<std::string> lines; //!< human explanation of an error

  /**
   * \brief Message default constructor
   * number is initialized to -1 as a flag for messages that have not
   * been defined yet.
   * other fields are default initialized to empty strings and vector.
   * This constructor is used when the vector of messages is created
   * or resized.
   */
  Message() : number(-1), severity('\0')
  {
  }

  /**
   * \brief fill in the elements of a message.
   * \param n the numeric code for the error message.
   * \param s the severity code.
   * \param m the rest of the macro text specifying the error message.
   * \return 0 if successful or 1 otherwise.
   *
   * Parse a symbolic message identifier or generate a symbolic name
   * from the message text.
   */
  int
  fill(int n, char s, std::string &m)
  {
    number = n;
    severity = s;
    // find the boundaries of a message text in quotes " or '.
    std::string::size_type begin_position = m.find_first_of("\"'");
    if (begin_position != std::string::npos) {
      std::string::size_type end_position =
          m.find_last_of(m.substr(begin_position, 1));
      if (end_position != std::string::npos) {
        // message text is the text in quotes excluding the quotes.
        message =
            m.substr(begin_position + 1, end_position - begin_position - 1);
        // find the symbol or generate from message if none found.
        begin_position = m.find_first_not_of(" \t", end_position + 1);
        std::ostringstream buffer;
        if (begin_position != std::string::npos)
          buffer << m.substr(begin_position);
        else
          buffer << severity << "_" << std::setfill('0') << std::setw(4)
                 << number << "_" << message;
        symbol = buffer.str();
        // replace $ with OPn, where n is 1, 2, ...
        int op = 1;
        bool more = true;
        while (more) {
          more = false;
          begin_position = symbol.find_first_of("$");
          if (begin_position != std::string::npos) {
            more = true;
            buffer.str("");
            buffer << "OP" << op++;
            symbol.replace(begin_position, 1, buffer.str());
          }
        }
        // replace illegal characters with '_' avoiding multiple
        // sequential underscores
        begin_position = 0;
        while (begin_position != std::string::npos) {
          begin_position = symbol.find_first_not_of(
              "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
              begin_position);
          if (begin_position != std::string::npos) {
            if (symbol[begin_position - 1] == '_')
              symbol.replace(begin_position, 1, "");
            else
              symbol.replace(begin_position++, 1, "_");
          }
        }
        // delete the trailing underscore, if any.
        begin_position = symbol.size() - 1;
        if (symbol[begin_position] == '_')
          symbol.replace(begin_position, 1, "");
        return 0;
      }
    }
    return 1;
  }
};

/**
 * \class Errmsg
 * \brief the application class encapsulates the application logic.
 */
class Errmsg
{

private:
  std::vector<const char *> input_filenames; //!< one or more input files
  const char *header_filename;               //!< c header output file
  const char *nroff_filename;                //!< aggregated doc output file
  const char *sphinx_filename;               //!< aggregated Sphinx file
  //! lines before the first message definition
  std::vector<std::string> initial_lines;
  std::vector<Message> messages; //!< recorded error messages

public:
  /**
   * \brief A constructor that processes the command line options.
   *
   * Multiple options of the same kind are ignored, only the last one
   * is used, except for input filename arguments, e.g.
   * $ ./errmsg -o out1.h -e err1.n in1.n in2.n -o out2.h
   *
   * ignores out1.h, reads in1.n and in2.n as input, writes C
   * definitions to out2.h, and writes aggregated nroff output to
   * err1.n.  Throws an exception if the command line options are
   * wrong or missing.
   */
  Errmsg(int argc, char *argv[])
      : header_filename(0), nroff_filename(0), sphinx_filename(0)
  {
    for (int arg = 1; arg < argc; ++arg) {
      if (strcmp(argv[arg], "-e") == 0) {
        if (++arg < argc)
          nroff_filename = argv[arg];
        else
          usage("missing error file name");
      } else if (strcmp(argv[arg], "-o") == 0) {
        if (++arg < argc)
          header_filename = argv[arg];
        else
          usage("missing output file name");
      } else if (strcmp(argv[arg], "-s") == 0) {
        if (++arg < argc)
          sphinx_filename = argv[arg];
        else
          usage("missing Sphinx file name");
      } else if (argv[arg][0] == '-')
        usage("unknown option");
      else
        input_filenames.push_back(argv[arg]);
    }
    if (input_filenames.size() == 0)
      usage("missing input file name");
    if (nroff_filename == 0)
      usage("missing nroff doc file name");
    if (header_filename == 0)
      usage("missing header file name");
  }

  /**
   * \brief The driver of the utility program.
   *  Read the input files and write the output files.
   * \return 0 if successful, 1 otherwise.
   */
  int
  run()
  {
    for (std::vector<const char *>::iterator it = input_filenames.begin();
         it != input_filenames.end(); ++it)
      if (read_input(*it))
        return 1;
    if (sphinx_filename && write_aggregated_sphinx())
      return 1;
    if (write_aggregated_nroff())
      return 1;
    return write_c_declarations();
  }

private:
  /**
   * \brief output usage information for the program
   * Exit if invalid command line options are given.
   * \param error the error messages indicating a problem with the
   * command line options.
   */
  void
  usage(const char *error = 0)
  {
    std::cout << "Usage: errmsg -e doc_file -o header_file [-s sphinx_file] "
                 "input_file(s)\n\n";
    std::cout
        << "input_file(s)  -- one or more input files with error messages\n";
    std::cout
        << "-e doc_file    -- file to write the aggregated nroff output\n";
    std::cout
        << "-o header_file -- file to write the aggregated C declarations\n";
    std::cout
        << "-s sphinx_file -- file to write the output in Sphinx format\n\n";
    if (error) {
      std::cerr << "Invalid command line: " << error << "\n\n";
      std::exit(1);
    }
  }

  /**
   * \brief read a single input file
   * Fill in the information for error messages defined in the input
   * file.
   * \param filename the name of the input file.
   * \return 0 if successful or 1 otherwise.
   */
  int
  read_input(const char *filename)
  {
    std::ifstream ifs(filename);
    if (!ifs) {
      std::cerr << "Input file could not be opened: " << filename << "\n";
      return 1;
    }
    int ret = 0;
    int lineno = 0;
    std::vector<std::string> *lines = &initial_lines;
    for (std::string line; std::getline(ifs, line); ++lineno) {
      std::vector<Message>::size_type num;
      std::string text;
      char severity;

      if (line.compare(0, 4, ".MS ")) {
        lines->push_back(line);
        continue;
      }
      // extract the components of a macro .MS W 109 "..."
      std::istringstream iss(line.substr(4));
      iss >> severity >> num;
      std::getline(iss, text);
      if (text.size() < 2) {
        std::cerr << filename << ":" << lineno << ": bad .MS macro: " << line
                  << "\n";
        ret = 1;
        continue;
      }
      if (num >= messages.size())
        messages.resize(num + 1);
      // check a messages with the same number has been recorded earlier.
      if (messages[num].number != -1) {
        std::cerr << filename << ":" << lineno
                  << ": two messages with the same number: " << num << "\n";
        ret = 1;
        continue;
      }
      // message text is anything between <"> and <"> or <'> and <'>.
      if (messages[num].fill(num, severity, text)) {
        std::cerr << filename << ":" << lineno << "message text missing: %d"
                  << num << "\n";
        ret = 1;
        continue;
      }
      lines = &(messages[num].lines);
      lines->push_back(line);
    }
    return ret;
  }

  /**
   * \brief get rid of surrounding quotation mark if any.
   * \param input a text that may have leading or trailing whitespace
   * and contained in quotation marks "text".
   * \return a new string with quotation marks stripped or the
   * input text if no quotation marks found.
   */
  std::string
  strip_surrounding_quotes(const std::string &input)
  {
    auto begin_pos = input.find_first_of('"');
    if (begin_pos == std::string::npos)
      return input;
    auto end_pos = input.find_first_of('"', begin_pos + 1);
    if (end_pos == std::string::npos)
      return input;
    return input.substr(begin_pos + 1, end_pos - begin_pos - 1);
  }

  /**
   * \brief convert a single line of text from NROFF to RST.
   * Parse NROFF control sequences and transform the text to suitable
   * RST representation that preserves typesetting encoded in NROFF.
   * \param input a string of text to be transformed.
   * \return transformed string.
   */
  std::string
  transform_nroff_to_sphinx(const std::string &input)
  {
    std::ostringstream oss;
    if (input[0] == '.') {
      std::istringstream iss(input.substr(1));
      std::string macro;
      iss >> macro;
      if (macro == "NS") {
        int chapter;
        iss >> chapter;
        std::string header;
        std::getline(iss, header);
        // extract only the first quoted part of the line.
        auto begin_pos = header.find_first_of('"');
        std::string::size_type end_pos;
        if (begin_pos == std::string::npos) {
          begin_pos = 0;
          end_pos = header.size() + 1;
        } else {
          begin_pos += 1;
          end_pos = header.find_first_of('"', begin_pos);
        }
        std::string underline(end_pos - begin_pos, '*');
        oss << underline << "\n"
            << header.substr(begin_pos, end_pos - begin_pos) << "\n"
            << underline;
      } else if (macro == "US") {
        std::string text;
        std::getline(iss, text);
        oss << "* " << strip_surrounding_quotes(text) << " - ";
      }
    } else
      oss << input;
    return oss.str();
  }

  /**
   * \brief escape the RST inline markup characters in text.
   * Replace special RST symbols with their escaped equivalents so
   * that these symbols appear in the typeset document as themselves.
   * \param input a text string to be transformed.
   * \return the input text string with all markup characters escaped.
   */
  std::string
  escape_markup(std::string &input)
  {
    for (std::string::size_type pos = 0; pos != std::string::npos;) {
      pos = input.find_first_of("*`|", pos);
      if (pos != std::string::npos) {
        // if a markup character is already escaped don't do anything,
        // because it's a part of an NROFF controlling sequence.
        if (pos == 0 || input[pos - 1] != '\\') {
          input.insert(pos, 1, '\\');
          pos += 2;
        } else {
          pos += 1;
        }
      }
    }
    return input;
  }

  /**
   * \brief trim any trailing whitespace.
   * \param input a string of text.
   * \return the input text without any trailing whitespace at the
   * end.
   */
  std::string
  trim_trailing_whitespace(const std::string &input)
  {
    std::string::size_type pos = input.find_last_not_of(' ');
    return input.substr(0, pos + 1);
  }

  /**
   * \brief write the aggregated Sphinx file with all error messages.
   *
   * \return 0 if successful or 1 otherwise.
   */
  int
  write_aggregated_sphinx()
  {
    std::ofstream out(sphinx_filename);
    if (!out) {
      std::cerr << "Output file could not be opened: " << sphinx_filename
                << "\n";
      return 1;
    }
    // output the initial lines.
    for (std::vector<std::string>::iterator it = initial_lines.begin();
         it != initial_lines.end(); ++it)
      out << transform_nroff_to_sphinx(*it) << "\n";
    // output each message.
    out << std::setfill('0');
    for (std::vector<Message>::iterator m = messages.begin();
         m != messages.end(); ++m) {
      if (m->number != -1) {
        out << "**" << m->severity << std::setw(3) << m->number << "** *"
            << trim_trailing_whitespace(m->message) << "*\n";
        for (std::vector<std::string>::iterator it = m->lines.begin() + 1;
             it != m->lines.end(); ++it)
          out << "   " << transform_nroff_to_sphinx(escape_markup(*it)) << "\n";
        out << "\n";
      }
    }
    return 0;
  }

  /**
   * \brief write the aggregated nroff file with all error messages.
   *
   * \return 0 if successful or 1 otherwise.
   */
  int
  write_aggregated_nroff()
  {
    std::ofstream out(nroff_filename);
    if (!out) {
      std::cerr << "Output file could not be opened: " << nroff_filename
                << "\n";
      return 1;
    }
    // output the initial lines.
    for (std::vector<std::string>::iterator it = initial_lines.begin();
         it != initial_lines.end(); ++it)
      out << *it << "\n";
    // output each message.
    for (std::vector<Message>::iterator m = messages.begin();
         m != messages.end(); ++m)
      for (std::vector<std::string>::iterator it = m->lines.begin();
           it != m->lines.end(); ++it)
        out << *it << "\n";
    return 0;
  }

  /**
   * \brief write the c header file with error message definitions.
   *
   * \return 0 if successful or 1 otherwise.
   */
  int
  write_c_declarations()
  {
    std::ofstream out(header_filename);
    if (!out) {
      std::cerr << "Output file could not be opened: " << header_filename
                << "\n";
      return 1;
    }
    out << "// This file written by utility program errmsg.  Do not modify.\n"
      "#ifdef ERRMSG_GET_ERRTXT_TABLE\n"
      "#ifndef ERRMSGDFa_H_\n"
      "#define ERRMSGDFa_H_\n"
      "static const char *errtxt[] = {\n";

    out << std::setfill('0');
    for (std::vector<Message>::size_type num = 0; num < messages.size();
         ++num) {
      out << " /* " << std::setw(3) << num << " */";
      if (messages[num].number == -1)
        out << " \"\",\n";
      else
        out << " \"" << messages[num].message << "\",\n";
    }
    out << "};\n"
      "#endif // ERRMSGDFa_H_\n"
      "#else /* ERRMSG_GET_ERRTXT_TABLE */\n"
      "#ifndef ERRMSGDFb_H_\n"
      "#define ERRMSGDFb_H_\n";
    // emit an enumeration of all the symbolic error codes.
    out << "enum error_code {\n";
    for (std::vector<Message>::iterator m = messages.begin();
         m != messages.end(); ++m) {
      if (m->number != -1) {
        out << "\n    " << m->symbol << " = " << m->number << ",\n";
        for (std::vector<std::string>::iterator it = m->lines.begin() + 1;
             it != m->lines.end(); ++it)
          out << "    // " << *it << "\n";
      }
    }
    out << "};\n"
      "#endif // ERRMSGDFb_H_\n"
      "#endif /* ERRMSG_GET_ERRTXT_TABLE */\n";
    return 0;
  }
};

/**
 * \brief the application's main function.
 *
 * Create an application class object and invoke its run() method.
 *
 * \param argc - argument count
 * \param argv - array of arguments
 */
int
main(int argc, char *argv[])
{
  Errmsg app(argc, argv);
  return app.run();
}
